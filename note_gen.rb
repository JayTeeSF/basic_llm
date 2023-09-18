#!/usr/bin/env ruby

require_relative "./full.rb"

full_cmd = './full.rb 2>&1'
full_out = %x(#{full_cmd})
grep_cmd = "grep 'transduction' corpus.txt"
grep_out = %x(#{grep_cmd})
cat_cmd = "cat full.rb"
cat_out = %x(#{cat_cmd})

enum_instructions = <<-EOF
The goal is to build an attention-based language model like chatGPT or Claude, where:
  -  Input Embedding Layer: Convert input words/tokens into numerical vector representations. This allows the model to understand semantic relationships between words.
  -  Self-Attention Layers: Relate different words in a sentence to each other, helping the model understand context. The core mechanism is computing compatibility scores between words and using those to aggregate information.
  -  Feedforward Layers: Process the intermediate representations from the attention layers to extract higher-level features.
  -  Output Layer: Convert the features from the last feedforward layer into predicted next-word probabilities.

  The key is stacking multiple self-attention and feedforward layers between the in put and output, so that we have a full model that can learn complex relationships between words and sentence structure.

  What's missing?
  We are currently missing key components needed for a full trainable model, including:

    -  A training loop that iterates through batches of input/target pairs
    -  An optimizer to update model weights like stochastic gradient descent
    -  A loss function to compare model outputs to targets like cross-entropy
    -  Logging of training loss over time to track progress

  To enable training, I would suggest adding:

    -  A TrainableModel class that encapsulates the training loop, optimizer, and loss calc
    -  An input pipeline using Dataset to load and batch the training data
    -  A training script that initializes the model, trains it, and evaluates results

  Be sure to add debug prints within the training loop to track metrics over time like loss and accuracy.
  Include a snippet of what an expected output should look like, so that anybody reviewing these debug statements will know (for the most common cases) what to do


Therefore, given our progress to-date, your remaining instructions are:
1. Please find and fix the bugs in the current full.rb code (see below). 
2. Complete any missing methods or method-logic.
3. If necessary, update the script that calls these classes to include a mode for training...
4. If anything is missing from the basic design (see below), or the training updates (see above) add it (i.e. missing classes or logic).
EOF

basic_design = <<-EOF
Here is the basic design:
```
Input Words         ->   Input Embedding Layer   -> Embedded Word Vectors

Embedded Vectors    ->   Self-Attention Layer    -> Contextual Vectors  

Context Vectors     ->   Feedforward Layer      -> Higher-Level Features

Higher-Level Features -> Output Layer       -> Next Word Probabilities
```
EOF

code_run = <<-EOF
Here is the output from running the code:
```
#{full_out}
```
EOF

corpus = <<-EOF
Here is a sample of my corpus.txt
`#{grep_cmd}`
```
#{grep_out}
```
EOF


codez = <<-EOF
Below is the current code:
```
#{cat_out}
```
EOF

# fail("trying to generate code_run: #{code_run}")

puts <<-EOF
#{enum_instructions}

#{basic_design}

#{corpus}

#{code_run}

#{codez}
EOF
