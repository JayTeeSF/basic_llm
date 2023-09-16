Here's a quick overview of how attention fits in:

The transformer takes a sequence of input tokens (like a sentence)
Each token goes through an initial embedding/projection layer to convert it to a "query", "key" and "value" vector
The attention layer then calculates scores between each query and all the keys
The values for the highest scored keys are summed to create the attention output for that query
This happens for each query token, so each token is "attended" to in this way

Some key points:

The attention scores allow the model to focus on the most relevant parts of the input
Multiple attention layers can be stacked to repeatedly refine the focus
The attention outputs are fed into feed-forward layers to make predictions

For training:

The whole model, including the attention layers, is trained end-to-end on some prediction task
So the attention learns to focus on the right parts of the input for that task
The attention weights are not trained separately or pre-defined
So in summary, attention is intertwined with the other components of a transformer to
bring relevant context into generating predictions from the input text.
The entire network is trained jointly in an end-to-end fashion for some supervised task.
