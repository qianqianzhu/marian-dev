#pragma once

// clang-format off
#include "graph/expression_graph.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"
#include "optimizers/optimizers.h"
#if MPI_FOUND
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#undef HOST
#define OMPI_SKIP_MPICXX 1 // Fixes compilation with GCC8+ https://github.com/open-mpi/ompi/issues/5157
#include "mpi.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#endif

//CCL
#include <oneapi/ccl.hpp>
// clang-format on

namespace marian {

struct/*interface*/ IMPIWrapper; // @TODO: Should we use a separate header, or move this declaration up here?

// This interface implements the cross-GPU operations for distributed training within a single box.
class ICommunicator {
protected:
  const std::vector<Ptr<ExpressionGraph>> graphs_;

public:
  ICommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs)
      : graphs_(graphs) {}

  virtual ~ICommunicator() {}

  // helper to apply a function to each local graph, in parallel threads
  typedef std::function<void(size_t, size_t /*shardBegin*/, size_t /*shardEnd*/)> ForeachFunc;
  virtual void foreach(const ForeachFunc& func, bool parallel = true) const = 0;
  // @TODO: We probably can still share foreach() between the two implementations. Just need to move some helper functions from the .cu file.

  virtual void scatterReduceAndResetGrads() const = 0; // reduce param gradients and scatter into gradient shards
  virtual void allGatherParams() const = 0;     // redistribute value shards into param values

  virtual void swapParams(const std::vector<Tensor>& paramShards) const = 0;

  virtual void scatterState(const std::vector<float>& data, const OptimizerBase::ScatterStateSetFunc& setFn) const = 0;
  virtual std::vector<float> gatherState(const OptimizerBase::GatherStateGetFunc& getFn) const = 0;
};

// Abstracts MPI operations, allowing alternative implementations (specifically fake (for debugging) and NCCL.
// This implements the MPI APIs we use here, with the following modifications:
//  * aborts with ABORT() instead of returning an error
//  * swapped out some strange MPI-specific data types to more correct C++ ones where appropriate
#if MPI_FOUND
#else
enum MPI_Comm { MPI_COMM_WORLD };
enum MPI_Datatype { MPI_FLOAT, MPI_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG, MPI_BYTE };
enum MPI_Op { MPI_SUM };
struct MPI_Status { int MPI_SOURCE; };
#define MPI_ANY_SOURCE ((size_t)-2)
#define MPI_STATUS_IGNORE ((MPI_Status*)nullptr)
#endif
struct/*interface*/ IMPIWrapper
{
  virtual size_t myMPIRank() const = 0;
  virtual size_t numMPIProcesses() const = 0;
  virtual void barrier(MPI_Comm comm = MPI_COMM_WORLD) const = 0;
  virtual void bCast(void* buf, size_t count, MPI_Datatype datatype, size_t rootRank = 0, MPI_Comm comm = MPI_COMM_WORLD) const = 0;
  virtual void sSend(void* buf, size_t count, MPI_Datatype datatype, size_t destRank, int tag, MPI_Comm comm = MPI_COMM_WORLD) const = 0;
  virtual void recv(void* buf, size_t count, MPI_Datatype datatype, size_t sourceRank, int tag, MPI_Comm comm = MPI_COMM_WORLD, MPI_Status* status = MPI_STATUS_IGNORE) const = 0;
  virtual void allReduce(const void* sendbuf, void* recvbuf, size_t count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) const = 0;
  virtual void finalize() = 0;
  static const size_t RECV_ANY_SOURCE = (size_t)MPI_ANY_SOURCE;
  // helper templates
private:
  static MPI_Datatype getDataType(const float*)              { return MPI_FLOAT; }
  static MPI_Datatype getDataType(const unsigned long*)      { return MPI_UNSIGNED_LONG; }
  static MPI_Datatype getDataType(const unsigned long long*) { return MPI_UNSIGNED_LONG_LONG; }
public:
  template<typename T>
  void bCast(std::vector<T>& v, size_t rootRank = 0, MPI_Comm comm = MPI_COMM_WORLD) {
    unsigned long long vecLen = (unsigned long long)v.size(); // only value from rootRank is used here
    bCast(&vecLen, 1, getDataType(&vecLen), rootRank, comm);
    v.resize(vecLen);
    bCast(v.data(), v.size(), getDataType(v.data()), rootRank, comm);
  }
  std::string idStr() const;
};

Ptr<IMPIWrapper> initMPI(bool multiThreaded);
void finalizeMPI(Ptr<IMPIWrapper>&&);

// DefaultCommunicator is used when we cannot use NCCLCommunicator, e.g. if it is not compiled in
class DefaultCommunicator : public ICommunicator {
private:
  std::vector<Ptr<TensorAllocator>> paramsAllocs_;
  std::vector<Tensor> tmpTensors_;

  void lazyInit() {
    if(tmpTensors_.size() == 0) {
      int totalSize = (int)graphs_[0]->params()->vals()->size();
      int shardSize = (int)ceil(totalSize / (float)graphs_.size());

      int pos = 0;
      for(auto graph : graphs_) {
        int __size__ = std::min(shardSize, totalSize);

        auto paramsAlloc = New<TensorAllocator>(graph->getBackend());
        paramsAllocs_.push_back(paramsAlloc);

        paramsAlloc->reserveExact(__size__ * sizeof(float));

        Tensor tmp;

        paramsAlloc->allocate(tmp, {1, __size__});
        tmpTensors_.push_back(tmp);

        // move to next shard
        pos += __size__;
        totalSize -= __size__;
      }
    }
  }

public:
  DefaultCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs, Ptr<IMPIWrapper> mpi)
      : ICommunicator(graphs) {
    ABORT_IF(mpi && mpi->numMPIProcesses() != 1, "DefaultCommunicator does not support multi-process MPI");
  }

  ~DefaultCommunicator() override {}

  void foreach(const ForeachFunc& func, bool parallel = true) const override {
    parallel &= graphs_.size() > 1;

    size_t totalSize = graphs_[0]->params()->vals()->size();
    size_t shardSize = (size_t)ceil(totalSize / (float)graphs_.size());

    size_t pos = 0;
    std::vector<std::thread> group;
    // iterate over all shards
    for(size_t idx = 0; idx < graphs_.size(); ++idx) {
      size_t size = std::min(shardSize, totalSize);

      if (parallel)
        group.emplace_back(func, idx, pos, pos+size);
      else
        func(idx, pos, pos+size);

      pos += size;
      totalSize -= size;
    }
    for(auto& t : group) // (note: group is empty is not parallel)
      t.join();
  }

  void scatterReduceAndResetGrads() const override {
    const_cast<DefaultCommunicator*>(this)->lazyInit();

    // Gather gradients from different devices into current gradient shards
    auto scatter = [this](size_t idx, size_t begin, size_t end) {
      auto curGrad = graphs_[idx]->params()->grads()->subtensor(begin, end-begin);

      // collect and sum gradients
      for(auto graph : graphs_) {
        if(graph != graphs_[idx]) {
          auto subGrad = graph->params()->grads()->subtensor(begin, end - begin);
          tmpTensors_[idx]->copyFrom(subGrad);

          using namespace functional;
          Element(_1 = _1 + _2, curGrad, tmpTensors_[idx]);
        }
      }
    };

    // reset gradients outside current shard
    auto reset = [this](size_t idx, size_t begin, size_t end) {
      auto grad = graphs_[idx]->params()->grads();
      if (begin > 0)
        grad->subtensor(0, begin)->set(0);
      if (end < grad->size())
        grad->subtensor(end, grad->size()-end)->set(0);
    };

    foreach(scatter);
    foreach(reset);
  }

  void allGatherParams() const override {

    // Update all graphs with parameter shard
    auto gather = [this](size_t idx, size_t begin, size_t end) {
      auto getShard = [&](Ptr<ExpressionGraph> graph) {
        return graph->params()->vals()->subtensor(begin, end-begin);
      };
      auto curShard = getShard(graphs_[idx]);

      // Copy parameter shard to each graph
      for(auto graph : graphs_) {
        if(graph != graphs_[idx]) {
          auto subShard = getShard(graph);
          subShard->copyFrom(curShard);
        }
      }
    };

    foreach(gather);
  }

  void swapParams(const std::vector<Tensor>& paramShards) const override {
    // Update all graphs with parameter shard
    auto gather = [this, paramShards](size_t idx, size_t begin, size_t end) {
      ABORT_IF(end - begin != paramShards[idx]->size(), "inconsistent shard size (swapParams, [{}], {} vs {})??", idx, end-begin, paramShards[idx]->size());
      // Copy parameter shard to each graph, apart from last graph
      for(int i = 0; i < (int)graphs_.size() - 1; ++i) {
        auto subParam = graphs_[i]->params()->vals()->subtensor(begin, paramShards[idx]->size());
        subParam->copyFrom(paramShards[idx]);
      }

      // Swap shard with corresponding share from last graph
      auto subParamLast = graphs_.back()->params()->vals()->subtensor(begin, paramShards[idx]->size());
      paramShards[idx]->swap(subParamLast);
    };
    // Execute for each shard
    foreach(gather);
  }

  void scatterState(const std::vector<float>& data, const OptimizerBase::ScatterStateSetFunc& setFn) const override {
    size_t dataSize = data.size();
    size_t numLocalDevices = graphs_.size();
    size_t shardSize = (dataSize + numLocalDevices - 1) / numLocalDevices;// (size_t)(ceil(dataSize / (float)numLocalDevices));
    for(size_t localDeviceIndex = 0; localDeviceIndex < numLocalDevices; localDeviceIndex++) {
      size_t begin = localDeviceIndex * shardSize;
      size_t end   = std::min(begin + shardSize, dataSize);
      setFn(localDeviceIndex, data.begin() + begin, data.begin() + end);
    }
  }

  std::vector<float> gatherState(const OptimizerBase::GatherStateGetFunc& getFn) const override {
    std::vector<float> data; // we know the size here
    for (size_t localDeviceIndex = 0; localDeviceIndex < graphs_.size(); localDeviceIndex++) {
      std::vector<float> tmp = getFn(localDeviceIndex);
      data.insert(data.end(), tmp.begin(), tmp.end());
    }
    ABORT_IF(data.size() != graphs_[0]->params()->vals()->size(), "gathering wrong amount of data??");
    return data;
  }
};


// This communicator should be used when on CPU with multiple nodes
class OneCCLCommunicator : public ICommunicator {
private:
  std::vector<Ptr<TensorAllocator>> paramsAllocs_;
  std::vector<Tensor> tmpTensors_;

  /*Copied from DefaultCommunicator*/
  void lazyInit() {
    if(tmpTensors_.size() == 0) {
      int totalSize = (int)graphs_[0]->params()->vals()->size();
      int shardSize = (int)ceil(totalSize / (float)graphs_.size());

      int pos = 0;
      for(auto graph : graphs_) {
        int __size__ = std::min(shardSize, totalSize);

        auto paramsAlloc = New<TensorAllocator>(graph->getBackend());
        paramsAllocs_.push_back(paramsAlloc);

        paramsAlloc->reserveExact(__size__ * sizeof(float));

        Tensor tmp;

        paramsAlloc->allocate(tmp, {1, __size__});
        tmpTensors_.push_back(tmp);

        // move to next shard
        pos += __size__;
        totalSize -= __size__;
      }
    }
  }
  /*These are copied from the NCCL codebase*/
  size_t myRank(size_t localDeviceIndex) const { // map local device index to a global rank
      return mpi_->myMPIRank() * graphs_.size() + localDeviceIndex;
  }

  size_t numRanks() const { // total number of devices across all MPI processes
      return mpi_->numMPIProcesses() * graphs_.size();
  }

  size_t dataSize() const { // total number of floats that comprise the concatenated parameter and gradient vector
    return graphs_[0]->params()->vals()->size();
  }

  // determine the (max) shard size
  // All shards except the last one have this size.
  // Presently, even all shards must have identical size
  size_t shardSize() const {
    size_t numShards = numRanks();
    size_t size = (dataSize() + numShards - 1) / numShards;
#if 1 // This is a good sanity check
    ABORT_IF(size * numShards != dataSize(), "presently, all shards must have the same size");
#endif
    return size;
  }

  // determine the index range (begin, end) of a shard
  std::pair<size_t, size_t> RankShardRange(size_t rank) const {
    size_t size = shardSize();
    size_t begin = rank * size;
    size_t end = begin + size;
    end = std::min(end, dataSize()); // clip last shard. Note: Presently this never happens, since shardSize() enforces uniform shard size.
    return{ begin, end };
  }

  // determine the index range (begin, end) of a shard
  std::pair<size_t, size_t> localShardRange(size_t localDeviceIndex) const {
    return RankShardRange(myRank(localDeviceIndex));
  }

  ccl::communicator commFactory(Ptr<IMPIWrapper> mpi) {
    ccl::init();

    int rank = mpi->myMPIRank();
    int size = mpi->numMPIProcesses();
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type kvs_addr;


    if (rank == 0) {
      kvs = ccl::create_main_kvs();
      kvs_addr = kvs->get_address();
      mpi->bCast((void*)kvs_addr.data(), ccl::kvs::address_max_size, MPI_BYTE, 0);
    } else {
      mpi->bCast((void*)kvs_addr.data(), ccl::kvs::address_max_size, MPI_BYTE, 0);
      kvs = ccl::create_kvs(kvs_addr);
    }
    std::cerr << "Creating comm" << std::endl;
    return  ccl::create_communicator(size, rank, kvs);
  }

public:
  ccl::communicator comm_;
  //std::vector<ccl::stream> streams_;
  Ptr<IMPIWrapper> mpi_; // Can not be null!
  OneCCLCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs, Ptr<IMPIWrapper> mpi)
      : ICommunicator(graphs),
        comm_(commFactory(mpi)),
        //streams_(graphs.size()),
        mpi_(mpi) {
      ABORT_IF(!mpi_, "We must have a valid MPI backend"); //We can't be null
      // Create one stream per communicator. Apparently there's no default stream constructor in this version.
      //for (int i=0; i< streams_.size(); i++) {
      //  streams_[i] = ccl::create_stream(); // Hopefully it's a CPU stream
      //}
  }

  ~OneCCLCommunicator() override {/*delete comm_;*/}

  /*Copied from default communicator. For whatever reason the NCCL communicator has a completely different implementation*/
  void foreach(const ForeachFunc& func, bool parallel = true) const override {
    std::cerr << "Entered foreach" << std::endl;
    parallel &= graphs_.size() > 1;
    parallel = false; //@TODO for some reason this just doesn't work for now

    size_t totalSize = graphs_[0]->params()->vals()->size();
    size_t shardSize = (size_t)ceil(totalSize / (float)graphs_.size());

    size_t pos = 0;
    std::vector<std::thread> group;
    // iterate over all shards
    for(size_t idx = 0; idx < graphs_.size(); ++idx) {
      size_t size = std::min(shardSize, totalSize);

      if (parallel)
        group.emplace_back(func, idx, pos, pos+size);
      else
        func(idx, pos, pos+size);

      pos += size;
      totalSize -= size;
    }
    for(auto& t : group) // (note: group is empty is not parallel)
      t.join();

    std::cerr << "Exitted foreach" << std::endl;
  }

  void scatterReduceAndResetGrads() const override {
    const_cast<OneCCLCommunicator*>(this)->lazyInit();

/* Old single node implementation
    // Gather gradients from different devices into current gradient shards
    auto scatter = [this](size_t idx, size_t begin, size_t end) {
      auto curGrad = graphs_[idx]->params()->grads()->subtensor(begin, end-begin);

      // collect and sum gradients
      for(auto graph : graphs_) {
        if(graph != graphs_[idx]) {
          auto subGrad = graph->params()->grads()->subtensor(begin, end - begin);
          tmpTensors_[idx]->copyFrom(subGrad);

          using namespace functional;
          Element(_1 = _1 + _2, curGrad, tmpTensors_[idx]);
        }
      }
    };*/
    std::cerr << "ScatterReduceAndReset" << std::endl;
    // We are all here;
    for(int i = 0; i < graphs_.size(); ++i) {
      ccl::stream stream = ccl::create_stream();
      ccl::barrier(comm_, stream);
      size_t begin, end; std::tie
      (begin, end) = localShardRange(i);

      auto grads = graphs_[i]->params()->grads();
      const auto* sendbuf = grads->data();
      auto*       recvbuf = grads->subtensor(begin, end-begin)->data();
      size_t      bufsize = shardSize();
      ABORT_IF(grads->subtensor(begin, end-begin)->size() != bufsize, "unexpected subtensor size??");
      //sendbuf, recvbuf, bufsize;
      //ABORT("Reduce_SCatter is not implemented yet");
      /*STUUUB  */
      ccl::reduce_scatter(sendbuf,
                          recvbuf,
                          bufsize,
                          ccl::reduction::sum,
                          comm_,
                          stream).wait();
                         
      ccl::barrier(comm_, stream);
    }

    // reset gradients outside current shard
    auto reset = [this](size_t idx, size_t begin, size_t end) {
      auto grad = graphs_[idx]->params()->grads();
      if (begin > 0)
        grad->subtensor(0, begin)->set(0);
      if (end < grad->size())
        grad->subtensor(end, grad->size()-end)->set(0);
    };

    //foreach(scatter);
    foreach(reset);
  }

  void allGatherParams() const override {

    /* Old implementation
    // Update all graphs with parameter shard
    auto gather = [this](size_t idx, size_t begin, size_t end) {
      auto getShard = [&](Ptr<ExpressionGraph> graph) {
        return graph->params()->vals()->subtensor(begin, end-begin);
      };
      auto curShard = getShard(graphs_[idx]);

      // Copy parameter shard to each graph
      for(auto graph : graphs_) {
        if(graph != graphs_[idx]) {
          auto subShard = getShard(graph);
          subShard->copyFrom(curShard);
        }
      }
    };
    */
    std::cerr << "AllGatherParams" << std::endl;
    for(int i = 0; i < graphs_.size(); ++i) {
      ccl::stream stream = ccl::create_stream();
      ccl::barrier(comm_, stream);
      size_t begin, end; std::tie
      (begin, end) = localShardRange(i);

      auto vals = graphs_[i]->params()->vals();
      const auto* sendbuf = vals->subtensor(begin, end-begin)->data();
      void*       recvbuf = vals->data();
      size_t      bufsize = shardSize();

      std::vector<size_t> counts(numRanks(), bufsize);

      ccl::allgatherv((const void *)sendbuf,
                      bufsize,
                      (void *)recvbuf,
                      counts,
                      ccl::datatype::float32,
                      comm_,
                      stream).wait();

      //NCCL_CHECK(ncclAllGather(sendbuf, recvbuf, bufsize, ncclFloat, comms_[i], streams_[i]));
      ccl::barrier(comm_, stream);
    }

    //foreach(gather);
  }
 /* Use NCCL's version
  void swapParams(const std::vector<Tensor>& paramShards) const override {
    // Update all graphs with parameter shard
    auto gather = [this, paramShards](size_t idx, size_t begin, size_t end) {
      ABORT_IF(end - begin != paramShards[idx]->size(), "inconsistent shard size (swapParams, [{}], {} vs {})??", idx, end-begin, paramShards[idx]->size());
      // Copy parameter shard to each graph, apart from last graph
      for(int i = 0; i < (int)graphs_.size() - 1; ++i) {
        auto subParam = graphs_[i]->params()->vals()->subtensor(begin, paramShards[idx]->size());
        subParam->copyFrom(paramShards[idx]);
      }

      // Swap shard with corresponding share from last graph
      auto subParamLast = graphs_.back()->params()->vals()->subtensor(begin, paramShards[idx]->size());
      paramShards[idx]->swap(subParamLast);
    };
    // Execute for each shard
    foreach(gather);
  }*/


    // swap distributed paramShards with model params()
  // It is assumed that all model params() on all devices and MPI processes are identical.
  // This is used for the smoothed parameters.
  void swapParams(const std::vector<Tensor>& distributedParamShards) const override {
    // get everything onto the CPU
    std::cerr << "SwapParams" << std::endl;
    auto distributedParams = gatherState([&](size_t localDeviceIndex) {
      std::vector<float> tmp;
      distributedParamShards[localDeviceIndex]->get(tmp);
      return tmp;
    });
    // Now all MPI processes hold an identical copy of a concatenation of all distributedParamShards[] across local and remote devices.
    std::vector<float> localParams;
    graphs_[0]->params()->vals()->get(localParams);
    // Now all MPI processes hold an identical copy of params() (remember, we assumed all devices hold the same params()).
    ABORT_IF(distributedParams.size() != localParams.size(), "distributed sharded and local params have different size??");

    // swap
    std::swap(distributedParams, localParams);

    // distribute it back
    scatterState(distributedParams, [&](size_t localDeviceIndex,
                                        std::vector<float>::const_iterator begin,
                                        std::vector<float>::const_iterator end){
      ABORT_IF(distributedParamShards[localDeviceIndex]->size() != end-begin, "swapParams size mismatch??"); // @TODO: move check to set()
      distributedParamShards[localDeviceIndex]->set(std::vector<float>(begin, end)); // @TODO: directly pass iterators to set()
    });
    for (auto& graph : graphs_) // broadcast to every local graph
      graph->params()->vals()->set(localParams);
  }

  // Collect shards across multiple devices and MPI processes in the NCCL configuration into a single CPU-side vector.
  // This is used when persisting optimizer state, which is sharded, and as part of swapParams().
  std::vector<float> gatherState(const OptimizerBase::GatherStateGetFunc& getFn) const override {
    std::cerr << "GatherState" << std::endl;
    std::vector<float> tmp; // (temp buffer used multiple times)
    // first, concatenate over all local devices
    std::vector<float> localData;
    for(size_t localDeviceIndex = 0; localDeviceIndex < graphs_.size(); localDeviceIndex++) {
      tmp = getFn(localDeviceIndex);
      localData.insert(localData.end(), tmp.begin(), tmp.end());
    }
    // second, concatenate across MPI processes
    // Note that all local devices occupy consecutive ncclRanks in order.
    std::vector<float> data;
    if (mpi_) {
      // push one rank's data at a time using broadcast
      for(size_t mpiRank = 0; mpiRank < mpi_->numMPIProcesses(); mpiRank++) {
        // broadcast mpiRank's localData to all
        if(mpiRank == mpi_->myMPIRank())
          tmp = localData;
        mpi_->bCast(tmp, /*rootRank=*/mpiRank);
        // now all ranks have the same slice: concatenate (we will end up with the same on all MPI processes)
        data.insert(data.end(), tmp.begin(), tmp.end());
      }
    }
    else { // no MPI: localData is the complete result already
      data = std::move(localData);
    }
    return data;
  }
/* Use NCCL's version instead
  void scatterState(const std::vector<float>& data, const OptimizerBase::ScatterStateSetFunc& setFn) const override {
    size_t dataSize = data.size();
    size_t numLocalDevices = graphs_.size();
    size_t shardSize = (dataSize + numLocalDevices - 1) / numLocalDevices;// (size_t)(ceil(dataSize / (float)numLocalDevices));
    for(size_t localDeviceIndex = 0; localDeviceIndex < numLocalDevices; localDeviceIndex++) {
      size_t begin = localDeviceIndex * shardSize;
      size_t end   = std::min(begin + shardSize, dataSize);
      setFn(localDeviceIndex, data.begin() + begin, data.begin() + end);
    }
  } */


  // Distribute a single CPU-side vector to shards across multiple devices and MPI processes.
  // This is used when restoring optimizer state, which is sharded, and as part of swapParams().
  // It is assumed that all MPI processes get the same data() passed. Hence, no MPI transfers are needed here.
  void scatterState(const std::vector<float>& data, const OptimizerBase::ScatterStateSetFunc& setFn) const override {
    std::cerr << "ScatterState" << std::endl;
    size_t dataSize = data.size();
    size_t numShards = numRanks();
    size_t shardSize = (dataSize + numShards - 1) / numShards;
    for(size_t localDeviceIndex = 0; localDeviceIndex < graphs_.size(); localDeviceIndex++) {
      // We only slice out data that is kept in our MPI process. Remember that all MPI processes receive the same, complete data.
      auto rank = myRank(localDeviceIndex);
      size_t begin = rank * shardSize;
      size_t end   = std::min(begin + shardSize, dataSize);
      setFn(localDeviceIndex, data.begin() + begin, data.begin() + end);
    }
  }

};

Ptr<ICommunicator> createCommunicator(
    const std::vector<Ptr<ExpressionGraph>>& graphs,
    bool noNccl, Ptr<IMPIWrapper> mpi);

}  // namespace marian
