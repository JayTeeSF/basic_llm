class Embeddings
  attr_reader :vocab
  def initialize(tokenizer, embedding_size: 300, vocab_size: 20_000)
    @embeddings = {}
    
    # Read corpus and tokenize
    # corpus = Tokenizer.encode(File.read(corpus_file) 
    # tokens = corpus.split
    # encodings = tokenizer.encode_file(corpus_file)
    encodings = tokenizer.encodings
    
    # Count frequency of each unique token
    freqs = Hash.new(0)
    encodings.each { |e| freqs[e] += 1 }
    
    # Keep most frequent as vocab
    @vocab = freqs.keys.sort_by{ |e| freqs[e] }.reverse[0..vocab_size]
    
    # Initialize embeddings to random values
    @vocab.each do |encoding|
      @embeddings[encoding] = Array.new(embedding_size) { rand } 
    end
  end
  
  def lookup(encoding)
    @embeddings[encoding]
  end
end
