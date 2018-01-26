#pragma once

#include "marian.h"

namespace marian {

class DecoderS2S_LM : public DecoderBase {
private:
  Ptr<rnn::RNN> rnn_;
  Ptr<rnn::RNN> rnn_LM; // Language model addition

  Ptr<rnn::RNN> constructLM(Ptr<ExpressionGraph> graph,
              Ptr<DecoderState> state) {

    float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");
    auto rnn = rnn::rnn(graph)                                     //
        ("type", opt<std::string>("dec-cell"))                     //
        ("dimInput", opt<int>("dim-emb"))                          //
        ("dimState", opt<int>("dim-rnn"))                          //
        ("dropout", dropoutRnn)                                    //
        ("layer-normalization", opt<bool>("layer-normalization"))  //
        ("nematus-normalization",
         options_->has("original-type")
             && opt<std::string>("original-type") == "nematus")  //
        ("skip", opt<bool>("skip"));

    size_t decoderLayers = opt<size_t>("dec-depth");
    size_t decoderBaseDepth = opt<size_t>("dec-cell-base-depth");
    size_t decoderHighDepth = opt<size_t>("dec-cell-high-depth");

    // setting up conditional (transitional) cell
    auto baseCell = rnn::stacked_cell(graph);
    for(int i = 1; i <= decoderBaseDepth; ++i) {
      bool transition = (i > 2);
      auto paramPrefix = prefix_ + "_lm_cell" + std::to_string(i);
      baseCell.push_back(rnn::cell(graph)         //
                         ("prefix", paramPrefix)  //
                         ("final", i > 1)         //
                         ("transition", transition)
                         ("fixed", !opt<bool>("trainable-interpolation")));
    }
    // Add cell to RNN (first layer)
    rnn.push_back(baseCell);

    // Add more cells to RNN (stacked RNN)
    for(int i = 2; i <= decoderLayers; ++i) {
      // deep transition
      auto highCell = rnn::stacked_cell(graph);

      for(int j = 1; j <= decoderHighDepth; j++) {
        auto paramPrefix
            = prefix_ + "_l" + std::to_string(i) + "_lm_cell" + std::to_string(j);
        highCell.push_back(rnn::cell(graph)("prefix", paramPrefix)("fixed", !opt<bool>("trainable-interpolation")));
      }

      // Add cell to RNN (more layers)
      rnn.push_back(highCell);
    }

    auto ret = rnn.construct();

    return ret;
  };

  Ptr<rnn::RNN> constructDecoderRNN(Ptr<ExpressionGraph> graph,
                                    Ptr<DecoderState> state) {
    float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");
    auto rnn = rnn::rnn(graph)                                     //
        ("type", opt<std::string>("dec-cell"))                     //
        ("dimInput", opt<int>("dim-emb"))                          //
        ("dimState", opt<int>("dim-rnn"))                          //
        ("dropout", dropoutRnn)                                    //
        ("layer-normalization", opt<bool>("layer-normalization"))  //
        ("nematus-normalization",
         options_->has("original-type")
             && opt<std::string>("original-type") == "nematus")  //
        ("skip", opt<bool>("skip"));

    size_t decoderLayers = opt<size_t>("dec-depth");
    size_t decoderBaseDepth = opt<size_t>("dec-cell-base-depth");
    size_t decoderHighDepth = opt<size_t>("dec-cell-high-depth");

    // setting up conditional (transitional) cell
    auto baseCell = rnn::stacked_cell(graph);
    for(int i = 1; i <= decoderBaseDepth; ++i) {
      bool transition = (i > 2);
      auto paramPrefix = prefix_ + "_cell" + std::to_string(i);
      baseCell.push_back(rnn::cell(graph)         //
                         ("prefix", paramPrefix)  //
                         ("final", i > 1)         //
                         ("transition", transition));
      if(i == 1) {
        for(int k = 0; k < state->getEncoderStates().size(); ++k) {
          auto attPrefix = prefix_;
          if(state->getEncoderStates().size() > 1)
            attPrefix += "_att" + std::to_string(k + 1);

          auto encState = state->getEncoderStates()[k];

          baseCell.push_back(rnn::attention(graph)  //
                             ("prefix", attPrefix)  //
                                 .set_state(encState));
        }
      }
    }
    // Add cell to RNN (first layer)
    rnn.push_back(baseCell);

    // Add more cells to RNN (stacked RNN)
    for(int i = 2; i <= decoderLayers; ++i) {
      // deep transition
      auto highCell = rnn::stacked_cell(graph);

      for(int j = 1; j <= decoderHighDepth; j++) {
        auto paramPrefix
            = prefix_ + "_l" + std::to_string(i) + "_cell" + std::to_string(j);
        highCell.push_back(rnn::cell(graph)("prefix", paramPrefix));
      }

      // Add cell to RNN (more layers)
      rnn.push_back(highCell);
    }

    return rnn.construct();
  }

public:
  DecoderS2S_LM(Ptr<Options> options) : DecoderBase(options) {}

  virtual Ptr<DecoderState> startStateLM(
      Ptr<ExpressionGraph> graph,
      Ptr<data::CorpusBatch> batch) {
    using namespace keywords;
    //Start state for LM deecoder.
    int dimBatch = batch->size();
    int dimRnn = opt<int>("dim-rnn"); //Assuming the same decoder shape as the one used
    Expr lm_start = graph->constant({dimBatch, dimRnn}, init = inits::zeros);

    rnn::States lm_startStates(opt<size_t>("dec-depth"), {lm_start, lm_start});

    std::vector<Ptr<EncoderState>> empty_encStates_;

    return New<DecoderState>(lm_startStates, nullptr, empty_encStates_);

  }

  virtual Ptr<DecoderState> startState(
      Ptr<ExpressionGraph> graph,
      Ptr<data::CorpusBatch> batch,
      std::vector<Ptr<EncoderState>>& encStates) {
    using namespace keywords;

    std::vector<Expr> meanContexts;
    for(auto& encState : encStates) {
      // average the source context weighted by the batch mask
      // this will remove padded zeros from the average
      meanContexts.push_back(weighted_average(
          encState->getContext(), encState->getMask(), axis = -3));
    }

    Expr start;
    if(!meanContexts.empty()) {
      // apply single layer network to mean to map into decoder space
      auto mlp = mlp::mlp(graph).push_back(
          mlp::dense(graph)                                          //
          ("prefix", prefix_ + "_ff_state")                          //
          ("dim", opt<int>("dim-rnn"))                               //
          ("activation", mlp::act::tanh)                             //
          ("layer-normalization", opt<bool>("layer-normalization"))  //
          ("nematus-normalization",
           options_->has("original-type")
               && opt<std::string>("original-type") == "nematus")  //
          );
      start = mlp->apply(meanContexts);
    } else {
      int dimBatch = batch->size();
      int dimRnn = opt<int>("dim-rnn");

      start = graph->constant({dimBatch, dimRnn}, init = inits::zeros);
    }

    rnn::States startStates(opt<size_t>("dec-depth"), {start, start});

    auto lm_decoderStates_ = startStateLM(graph, batch);

    return New<DecoderState>(startStates, nullptr, encStates, lm_decoderStates_);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) {
    using namespace keywords;

    auto embeddings = state->getTargetEmbeddings();

    auto lm_embeddings = state->getLMState()->getTargetEmbeddings();

    // dropout target words
    float dropoutTrg = inference_ ? 0 : opt<float>("dropout-trg");
    if(dropoutTrg) {
      int trgWords = embeddings->shape()[-3];
      auto trgWordDrop = graph->dropout(dropoutTrg, {trgWords, 1, 1});
      embeddings = dropout(embeddings, mask = trgWordDrop);
      lm_embeddings = dropout(lm_embeddings, mask = trgWordDrop); //@TODO is that the lm_dropout
    }

    if(!rnn_)
      rnn_ = constructDecoderRNN(graph, state);

    if(!rnn_LM)
      rnn_LM = constructLM(graph, state->getLMState());

    // apply RNN to embeddings, initialized with encoder context mapped into
    // decoder space
    auto decoderContext = rnn_->transduce(embeddings, state->getStates());

    // apply RNN to embeddings to LM, initialized with encoder context mapped into
    // decoder space
    auto lm_decoderContext = rnn_LM->transduce(lm_embeddings, state->getLMState()->getStates());

    // retrieve the last state per layer. They are required during translation
    // in order to continue decoding for the next word
    rnn::States decoderStates = rnn_->lastCellStates();

    // retrieve the last LM state per layer. They are required during translation
    // in order to continue decoding for the next word
    rnn::States lm_decoderStates = rnn_LM->lastCellStates();

    std::vector<Expr> alignedContexts; //@TODO this empty for the LM?
    for(int k = 0; k < state->getEncoderStates().size(); ++k) {
      // retrieve all the aligned contexts computed by the attention mechanism
      auto att = rnn_->at(0)
                     ->as<rnn::StackedCell>()
                     ->at(k + 1)
                     ->as<rnn::Attention>();
      alignedContexts.push_back(att->getContext());
    }

    Expr alignedContext;
    if(alignedContexts.size() > 1)
      alignedContext = concatenate(alignedContexts, axis = -1);
    else if(alignedContexts.size() == 1)
      alignedContext = alignedContexts[0];

    // construct deep output multi-layer network layer-wise
    auto layer1 = mlp::dense(graph)                                //
        ("prefix", prefix_ + "_ff_logit_l1")                       //
        ("dim", opt<int>("dim-emb"))                               //
        ("activation", mlp::act::tanh)                             //
        ("layer-normalization", opt<bool>("layer-normalization"))  //
        ("nematus-normalization",
         options_->has("original-type")
             && opt<std::string>("original-type") == "nematus");

    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];

    auto layer2 = mlp::dense(graph)           //
        ("prefix", prefix_ + "_ff_logit_l2")  //
        ("dim", dimTrgVoc);

    if(opt<bool>("tied-embeddings") || opt<bool>("tied-embeddings-all")) {
      std::string tiedPrefix = prefix_ + "_Wemb";
      if(opt<bool>("tied-embeddings-all") || opt<bool>("tied-embeddings-src"))
        tiedPrefix = "Wemb";
      layer2.tie_transposed("W", tiedPrefix);
    }

    // construct deep output multi-layer network layer-wise for the LM
    auto output = mlp::mlp(graph)         //
                      .push_back(layer1)  //
                      .push_back(layer2);

        // construct deep output multi-layer network layer-wise
    auto lm_layer1 = mlp::dense(graph)                                //
        ("prefix", prefix_ + "_lm_ff_logit_l1")                       //
        ("dim", opt<int>("dim-emb"))                               //
        ("activation", mlp::act::tanh)                             //
        ("layer-normalization", opt<bool>("layer-normalization"))  //
        ("nematus-normalization",
         options_->has("original-type")
             && opt<std::string>("original-type") == "nematus")
        ("fixed", !opt<bool>("trainable-interpolation"));


    auto lm_layer2 = mlp::dense(graph)           //
        ("prefix", prefix_ + "_lm_ff_logit_l2")  //
        ("dim", dimTrgVoc)
        ("fixed", !opt<bool>("trainable-interpolation"));

    if(opt<bool>("tied-embeddings") || opt<bool>("tied-embeddings-all")) {
      std::string tiedPrefix = prefix_ + "_lm_Wemb";
      //if(opt<bool>("tied-embeddings-all") || opt<bool>("tied-embeddings-src"))
      //  tiedPrefix = "_lm_Wemb"; We don't have source tied embeddings for the LM
      lm_layer2.tie_transposed("W", tiedPrefix);
    }

    // assemble layers into MLP and apply to embeddings, decoder context and
    // aligned source context
    auto lm_output = mlp::mlp(graph)         //
                      .push_back(lm_layer1)  //
                      .push_back(lm_layer2);


    Expr logits;
    Expr lm_logits;
    //Interpolation code begins here
    if(opt<std::string>("interpolation-type") == "shallow") {
      if(alignedContext)
        logits = output->apply(embeddings, decoderContext, alignedContext);
      else
        logits = output->apply(embeddings, decoderContext);


      lm_logits = lm_output->apply(lm_embeddings, lm_decoderContext);
      //Shallow interpolations with a hyperparameter
      logits = logits + 0.2*lm_logits;
    }else if(opt<std::string>("interpolation-type") == "shallow-trainable") {
      if(alignedContext)
        logits = output->apply(embeddings, decoderContext, alignedContext);
      else
        logits = output->apply(embeddings, decoderContext);


      lm_logits = lm_output->apply(lm_embeddings, lm_decoderContext);
      //Shallow interpolations, add a param here that is trainable that interpolates logits with lm_logits
      auto alpha = graph->param("alpha", {1,1}, keywords::init=inits::from_value(0.2));
      logits = logits + alpha*lm_logits;
    }else if(opt<std::string>("interpolation-type") == "shallow-vector") {
      if(alignedContext)
        logits = output->apply(embeddings, decoderContext, alignedContext);
      else
        logits = output->apply(embeddings, decoderContext);


      lm_logits = lm_output->apply(lm_embeddings, lm_decoderContext);
      //Shallow interpolations, add a param here that is trainable that interpolates logits with lm_logits
      //@TODO ask marcin how you multiply 2d by 3d
      //@TODO shape shows() that we produce what I think we do
      auto alpha = graph->param("alpha", {1, dimTrgVoc}, keywords::init=inits::from_value(0.2));
      logits = logits + alpha*lm_logits;
    }else if(opt<std::string>("interpolation-type") == "deep") {
      if(alignedContext)
        logits = output->apply(embeddings, decoderContext, alignedContext, lm_decoderContext);
      else
        logits = output->apply(embeddings, decoderContext, lm_decoderContext);

      //@TODO this is not necessary right
      lm_logits = lm_output->apply(lm_embeddings, lm_decoderContext);
    }else if(opt<std::string>("interpolation-type") == "extra-output") {
        /*
      auto lm_layer1 = mlp::dense(graph)                                //
        ("prefix", prefix_ + "_lm_ff_logit_l1")                       //
        ("dim", opt<int>("dim-emb"))                               //
        ("activation", mlp::act::tanh)                             //
        ("layer-normalization", opt<bool>("layer-normalization"))  //
        ("nematus-normalization",
         options_->has("original-type")
             && opt<std::string>("original-type") == "nematus")
        ("fixed", true);

        */
      auto interpolate_layer = mlp::dense(graph)           //
        ("prefix", prefix_ + "_lm_interpolate")
        ("activation", mlp::act::tanh)  //
        ("dim", dimTrgVoc);

       // assemble layers into MLP and apply to embeddings, decoder context and
       // aligned source context
       auto lm_interpolate = mlp::mlp(graph)         //
                        .push_back(interpolate_layer );  //
//                        .push_back(lm_layer2);

      if(alignedContext)
        logits = output->apply(embeddings, decoderContext, alignedContext);
      else
        logits = output->apply(embeddings, decoderContext);

      //@TODO this is not necessary right
      lm_logits = lm_output->apply(lm_embeddings, lm_decoderContext);
      auto lm_interpolate_logits = lm_interpolate->apply(embeddings, decoderContext, alignedContext);
      logits = logits + lm_interpolate_logits*lm_logits;
    }else {
      ABORT("Unknown interpolation type: {}", opt<std::string>("interpolation-type"));
    }

    auto lm_decoderStates_ = New<DecoderState>(lm_decoderStates, lm_logits, state->getLMState()->getEncoderStates());
      
    // return unormalized(!) probabilities
    return New<DecoderState>(decoderStates, logits, state->getEncoderStates(), lm_decoderStates_);
  }

  // helper function for guided alignment
  virtual const std::vector<Expr> getAlignments(int i = 0) {
    auto att
        = rnn_->at(0)->as<rnn::StackedCell>()->at(i + 1)->as<rnn::Attention>();
    return att->getAlignments();
  }

  void clear() {
    rnn_ = nullptr;
    rnn_LM = nullptr;
   }
};
}
