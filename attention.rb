require_relative './tokenizer'
class Attention
  # initialize with an array of input words (i.e. the embedding vector)
  def initialize(vocab_tok_str_array, tokenizer)
    @vocab_tok_str_array = vocab_tok_str_array
    # warn("@vocab_tok_str_array: #{@vocab_tok_str_array.inspect}")
    @tokenizer = tokenizer # || Tokenizer.new(vocab_tok_str_array)
  end

  # input: a single tokenized word (String)
  # not a Numeric token_id value
  def attend(query_token_str)
    query_token, _ = @tokenizer.encode([query_token_str])

    scores = {}
    @tokenizer.encode(@vocab_tok_str_array).each do |token|
      score = compute_score(query_token, token)
      puts("encoded token(#{token.inspect}) scored: #{score.inspect}")
      scores[token] = score
    end

    weights = softmax(scores)

    output = {}
    scores.each do |token, _score|
      output[token] = weights[token] * token # what is the meaning of this ?!
    end

    return output
  end

  private

  def compute_score(query_token_str, token)
    query_token_str == token ? 1.0 : 0.1
  end

  def softmax(scores)
    totals = scores.values.sum
    weights = {}

    scores.each do |token, score|
      weights[token] = Float(score) / totals
    end

    weights
  end
end
