#include "training/graph_group_async.h"

#include "kernels/tensor_operators.h"

namespace marian {

void AsyncGraphGroup::setScheduler(Ptr<Scheduler> scheduler) {
  scheduler_ = scheduler;
  // optimizer has to be registered last to see changes of learning rate
  scheduler_->registerTrainingObserver(scheduler_);

  for(auto opt : shardOpt_)
    scheduler_->registerTrainingObserver(opt);
}

void AsyncGraphGroup::fetchParams(Tensor oldParams,
                                  const std::vector<Tensor>& params,
                                  int device_id) {
  // @TODO read guard on parameters
  int pos = 0;

  std::vector<std::thread> threads;
  for(int idx = 0; idx < devices_.size(); idx++) {
    threads.emplace_back(std::thread(
        [&](int idx, int pos) {
          // individual mutex per-shard
          std::lock_guard<std::mutex> guard(shardSync_[idx]);
          oldParams->subtensor(pos, params[idx]->size())->copyFrom(params[idx]);
        },
        idx,
        pos));

    pos += shardSize_;
  }
  for(auto&& t : threads) {
    t.join();
  }
}

void AsyncGraphGroup::reversefetchParamsLocal(Tensor newParams, const std::vector<Tensor>& params, int idx) {
  int pos = shardSize_*idx;
  params[idx]->copyFrom(newParams->subtensor(pos, params[idx]->size()));
}

void AsyncGraphGroup::pushGradients(Tensor newGrads,
                                    size_t batch_words,
                                    int device_id) {
  // add instead of copy?
  std::vector<std::thread> threads;
  int pos = 0;
  for(int idx = 0; idx < devices_.size(); idx++) {
    threads.emplace_back(std::thread(
        [&](int idx, int pos) {
          // individual mutex per-shard
          std::lock_guard<std::mutex> guard(shardSync_[idx]);

          grads_[idx]->copyFrom(newGrads->subtensor(pos, grads_[idx]->size()));

          if (gradientBufferSize_ > 1) {
            using namespace functional;
            Element(_1 = _1 + _2, bufferGrads_[idx], grads_[idx]);
            bufferCount[idx]++;

            if (bufferCount[idx] < gradientBufferSize_) return;
            
            bufferCount[idx] = 0;
          }

          Tensor updater = (gradientBufferSize_ > 1)?bufferGrads_[idx]:grads_[idx];

          if(scaleLearningRate_) {
            shardOpt_[idx]->update(
                params_[idx], updater, batch_words / avgBatchWords_);
          } else {
            shardOpt_[idx]->update(params_[idx], updater);
          }

          if (gradientBufferSize_ > 1) bufferGrads_[idx]->set(0);

          if(movingAvg_)
            updateMovingAverage(
                paramsAvg_[idx], params_[idx], scheduler_->numberOfBatches());
        },
        idx,
        pos));

    pos += shardSize_;
  }
  for(auto&& t : threads)
    t.join();
}

void AsyncGraphGroup::updateMovingAverage(Tensor paramsAvg,
                                          Tensor params,
                                          size_t batches) {
  using namespace functional;
  float decay
      = std::max(mvDecay_, 1.f - (float)(batches + 1) / (float)(batches + 10));
  Element(_1 = ((1.f - decay) * _1) + (decay * _2), paramsAvg, params);
}

void AsyncGraphGroup::init(Ptr<data::Batch> batch) {
  // initialize the parameters
  for(size_t i = 0; i < graphs_.size(); ++i) {
    // takes care of thead_local stuff
    THREAD_GUARD(builders_[i]->build(graphs_[i], batch);
                 graphs_[i]->forward(););
  }

  if(params_.size() == 0) {
    int totalSize = graphs_[0]->params()->vals()->size();
    shardSize_ = ceil(totalSize / (float)devices_.size());

    int pos = 0;
    // parameter sharding
    for(auto device : devices_) {

      bufferCount.push_back(0);

      int __size__ = min(shardSize_, totalSize);
      totalSize -= __size__;

      Tensor param;
      Ptr<TensorAllocator> allocator = New<TensorAllocator>(device);
      allocator->reserveExact(__size__ * sizeof(float));
      allocator->allocate(param, {1, __size__});
      paramsAlloc_.push_back(allocator);

      param->copyFrom(graphs_[0]->params()->vals()->subtensor(pos, __size__));
      params_.push_back(param);

      pos += __size__;
    }
  }
  if(grads_.size() == 0) {
    int totalSize = graphs_[0]->params()->vals()->size();

    for(auto device : devices_) {
      int __size__ = min(shardSize_, totalSize);
      totalSize -= __size__;
      Tensor grad_;
      Ptr<TensorAllocator> allocator_ = New<TensorAllocator>(device);

      allocator_->reserveExact(__size__ * sizeof(float));
      allocator_->allocate(grad_, {1, __size__});
      gradsAlloc_.push_back(allocator_);
      grads_.push_back(grad_);

      
      if (gradientBufferSize_ > 1){
        Ptr<TensorAllocator> bufferAlloc = New<TensorAllocator>(device);
        Tensor bufferGrad_;

        bufferAlloc->reserveExact(__size__ * sizeof(float));
        bufferAlloc->allocate(bufferGrad_, {1, __size__});
        
        gradsAlloc_.push_back(bufferAlloc);
        bufferGrad_->set(0);
        bufferGrads_.push_back(bufferGrad_);
      }
    }
  }
  if(movingAvg_) {
    if(paramsAvg_.size() == 0) {
      int totalSize = graphs_[0]->params()->vals()->size();

      int i = 0;
      for(auto device : devices_) {
        int __size__ = min(shardSize_, totalSize);
        totalSize -= __size__;
        Tensor paramAvg;
        Ptr<TensorAllocator> allocator = New<TensorAllocator>(device);

        allocator->reserveExact(__size__ * sizeof(float));
        allocator->allocate(paramAvg, {1, __size__});

        paramAvg->copyFrom(params_[i++]);

        paramsAllocAvg_.push_back(allocator);
        paramsAvg_.push_back(paramAvg);
      }
    }
  }
}

void AsyncGraphGroup::execute(Ptr<data::Batch> batch) {
  if(first_) {
    init(batch);
    first_ = false;
  }

  auto task = [this](Ptr<data::Batch> batch) {
    static size_t i = 0;
    thread_local Ptr<ExpressionGraph> graph;
    thread_local Ptr<models::ModelBase> builder;
    thread_local size_t t = 0;
    thread_local size_t num_seen_words = 0;
    thread_local int t_id = 0;
    thread_local float cost = 0;
    thread_local int sentences = 0;
    thread_local int words = 0;
    thread_local float max_batch_words = 1920.0;
    thread_local float current_batch_words = 7200.0;

    thread_local Tensor accGradients;
    thread_local Ptr<TensorAllocator> accAlloc;

    //Thread local optimizer:
//    thread_local Ptr<OptimizerBase> localOpt = Ptr<OptimizerBase>(new Adam(options_->get<double>("learn-rate")));

//    if (t==0) {
//      localOpt->setB1(0.93f);
//      localOpt->setB1(0.997f);
//    }
    if (current_batch_words > 1920) {
    	current_batch_words = current_batch_words - 1.32;
        avgBatchWords_ = current_batch_words;
    }

    if(!graph) {
      std::lock_guard<std::mutex> lock(sync_);
      t_id = i;
      graph = graphs_[i];
      builder = builders_[i++];
    }

    auto costNode = builder->build(graph, batch);

    if(t % tau_ == 0) {
      fetchParams(graph->params()->vals(), params_, t_id);
    }

    graph->forward();
    cost += costNode->scalar();
    graph->backward();

    size_t batch_words = batch->words();
    
    words += batch->words();
    sentences += batch->size();

    // Get batch stats
    if (tau_ > 0 && t < local_optimizer_duration_) {
      localOpt->update(graph, batch_words/avgBatchWords_);
//      reversefetchParamsLocal(graph->params()->vals(),
//                    params_, t_id);
      //shardOpt_[my_id]->updateState(localOpt, shardSize_, my_id);

    }
    //Get batch stats
//    if (t == 4000) {
//      shardOpt_[t_id]->setB1(0.92);
//      shardOpt_[t_id]->setB2(0.998);
//    }


    Tensor gradients;
    if(tau_ > 1) {
      if(t == 0) {
        accAlloc = New<TensorAllocator>(graph->getDevice());
        accAlloc->reserveExact(graph->params()->grads()->memory()->size());
        accAlloc->allocate(accGradients, graph->params()->grads()->shape());
        accGradients->set(0);
      }

      using namespace functional;
      Element(_1 += _2, accGradients, graph->params()->grads());
      gradients = accGradients;

      // Keep track of how many words we've calculated the error from
      num_seen_words += batch_words;
    } else {
      gradients = graph->params()->grads();
      num_seen_words = batch_words;
    }

    t++;

    if(t % tau_ == 0) {
      pushGradients(gradients, num_seen_words, t_id);
      // Reset the counter of seen words after gradient update
      num_seen_words = 0;

      if(tau_ > 1)
        gradients->set(0);
    }

    if(t % (tau_ * gradientBufferSize_) == 0 && scheduler_) {
      std::unique_lock<std::mutex> lock(schedulerMutex_);

      // Wait until the thread that wants to do validation is finished.
      pool_.wait_for_one(lock);
      
      cost /= (tau_ * gradientBufferSize_);

      scheduler_->update(cost, sentences, words);
      sentences = 0;
      words = 0;
      cost = 0;

      if(scheduler_->saving() || scheduler_->validating()) {
        // Wait with validation or saving until all other threads are done with update.
        // We want to reuse the graphs for validation, so they need to be in
        // a safe state.
        pool_.wait_for_others(lock);

        if(movingAvg_)
          for(auto g : graphs_)
            fetchParams(g->params()->vals(), paramsAvg_, t_id);

        if(scheduler_->saving())
          this->save(graph);

        if(scheduler_->validating())
          scheduler_->validate(graphs_);

        // Validation or saving is done, tell other threads to continue work.
        pool_.notify_others();
      }
    }
  };

  pool_.enqueue(task, batch);
}
}
