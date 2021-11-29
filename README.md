# Sequence2Sequence-Implementation

The Seq2Seq framework relies on the encoder-decoder paradigm. The encoder encodes the input sequence, while the decoder produces the target sequence

### Encoder

Our input sequence is how are you. Each word from the input sequence is associated to a vector  𝑤∈ℝ𝑑  (via a lookup table). In our case, we have 3 words, thus our input will be transformed into  [𝑤0,𝑤1,𝑤2]∈ℝ𝑑×3 . Then, we simply run an LSTM over this sequence of vectors and store the last hidden state outputed by the LSTM: this will be our encoder representation  𝑒 . Let’s write the hidden states  [𝑒0,𝑒1,𝑒2]  (and thus  𝑒=𝑒2 )

## Decoder

Now that we have a vector  𝑒  that captures the meaning of the input sequence, we’ll use it to generate the target sequence word by word. Feed to another LSTM cell:  𝑒  as hidden state and a special start of sentence vector  𝑤𝑠𝑜𝑠  as input. The LSTM computes the next hidden state  ℎ0∈ℝℎ . Then, we apply some function  𝑔:ℝℎ↦ℝ𝑉  so that  𝑠0:=𝑔(ℎ0)∈ℝ𝑉  is a vector of the same size as the vocabulary.

ℎ0𝑠0𝑝0𝑖0=LSTM(𝑒,𝑤𝑠𝑜𝑠)=𝑔(ℎ0)=softmax(𝑠0)=argmax(𝑝0)
 
Then, apply a softmax to  𝑠0  to normalize it into a vector of probabilities  𝑝0∈ℝ𝑉  . Now, each entry of  𝑝0  will measure how likely is each word in the vocabulary. Let’s say that the word “comment” has the highest probability (and thus  𝑖0=argmax(𝑝0)  corresponds to the index of “comment”). Get a corresponding vector  𝑤𝑖0=𝑤𝑐𝑜𝑚𝑚𝑒𝑛𝑡  and repeat the procedure: the LSTM will take  ℎ0  as hidden state and  𝑤𝑐𝑜𝑚𝑚𝑒𝑛𝑡  as input and will output a probability vector  𝑝1  over the second word, etc.

ℎ1𝑠1𝑝1𝑖1=LSTM(ℎ0,𝑤𝑖0)=𝑔(ℎ1)=softmax(𝑠1)=argmax(𝑝1)
 
The decoding stops when the predicted word is a special end of sentence token.
