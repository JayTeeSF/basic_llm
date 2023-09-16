#!/usr/bin/env ruby

require_relative './attention'
require_relative './embeddings'

if __FILE__ == $PROGRAM_NAME
  corpus_path = './corpus.txt'
  warn(%|tmp_tokenizer = Tokenizer.from_file(#{corpus_path.inspect})...|)
  tmp_tokenizer = Tokenizer.from_file(corpus_path)

  warn(%|embeddings = Embeddings.new(tmp_tokenizer)...|)
  embeddings = Embeddings.new(tmp_tokenizer)
  tokenizer = Tokenizer.new(tmp_tokenizer.decode(embeddings.vocab))

  input_str = "What is multi-head self-attention"
  expected_decoded_inputs = ["What", "is", "multi-head", "self-attention"]
  inputs = Tokenizer.tokenize(input_str)
  warn(%|inputs(#{inputs.inspect}) ?= #{expected_decoded_inputs.inspect}: #{(inputs == expected_decoded_inputs).inspect}\n|)

  warn(%|attention = Attention.new(#{inputs.inspect}, tokenizer)...|)
  attention = Attention.new(inputs, tokenizer) # why strings? why not encoded tokens
  #warn(%|attention = Attention.new(#{input_str}, tokenizer)...|)
  #attention = Attention.new(input_str, tokenizer) # why strings? why not encoded tokens

  query = "animal"
  warn(%|output = attention.attend("#{query}")...|)
  output = attention.attend(query)

  warn("output: #{output.inspect}")

  string_output = {}
  output.each do |token, score|
    string_output["#{token} (#{tokenizer.decode([token]).first})"] = score
  end

  pp string_output
end
