import pretty_midi
import collections
import pandas as pd
import numpy as np
import gensim
import sentencepiece as spm
from joblib import load


__model = None

def load_model():
    global __model

    if __model is None:
        __model = load('/static/rf_model.joblib')

    print('Model Loading Successful')


def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        notes['velocity'].append(note.velocity)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def extract_gram(midi_frame):
    gram_list = []
    temp = []
    s_time = 0
    for i in range(midi_frame.shape[0]):
        pitch = midi_frame["pitch"][i]
        ti = round(midi_frame["duration"][i],2)
        gram = (pitch,ti)

        if (not temp) or (midi_frame["start"][i] - s_time <= 0.003):
            temp.append(gram)
            if(len(temp) == 1):
                s_time = midi_frame["start"][i]
            if(i == midi_frame.shape[0] - 1):
                gram_list += temp
        else:
            sorted_list = sorted(temp, key=lambda tup: tup[0], reverse=True)
            sorted_list.append(gram)
            gram_list += sorted_list
            temp.clear()
            s_time = 0

    return gram_list

def encodeChinese(index_number):
  val = index_number + 0x4e00
  return chr(val)

def get_sentence_vec_avg(sentences,model):
  l = []
  for sentence in sentences:
    for word in sentence:
      try:
        temp = np.zeros(len(model.wv[word]))
        temp += model.wv[word]
      except:
        print("Not in vocab")
    l.append(temp/len(sentence))
  return l

def get_sentence_vec_avg_with_cov2(sentences,model):
  l = []
  cov = []
  for sentence in sentences:
    for word in sentence:
      try:
        temp = np.zeros(len(model.wv[word]))
        temp += model.wv[word]
        cov.append(model.wv[word])
      except:
        print("Not in vocab")
    data = np.array(cov)
    sd = np.std(data,axis=0)
    z = temp/len(sentence)
    z = z.tolist()
    z += sd.tolist()
    z = np.array(z)
    l.append(z)
  return l

def get_sentence_vec_SD_only(sentences,model):
  l = []
  cov = []
  for sentence in sentences:
    for word in sentence:
      try:
        cov.append(model.wv[word])
      except:
        print("Not in vocab")
    data = np.array(cov)
    sd = np.std(data,axis=0)
    z = sd.tolist()
    z = np.array(z)
    l.append(z)
  return l


def sentencePiece(corpus, modelName, vocabSize, maxSenLength):

    spm.SentencePieceTrainer.train(input=corpus, model_prefix=modelName, vocab_size=vocabSize,
                                   max_sentence_length=maxSenLength)

    # Load trained model
    sp = spm.SentencePieceProcessor()
    model_path = modelName + ".model"
    sp.load(model_path)

    # Tokenize the single line
    f1 = open(corpus, 'r')
    line = f1.readline()
    tokenized = sp.encode_as_pieces(line)

    # Create a Series with a single entry
    Ch_note_series = pd.Series({0: tokenized})

    # Convert Series to a list of sentences
    sentences = Ch_note_series.tolist()

    return sentences


def Word2Vec(Window, sentences, Avg = False, SD = False):
  model = gensim.models.Word2Vec(
    sentences=sentences,
    window=Window,
    min_count=1,
    workers=4,
    sg = 1
  )
  if(Avg and SD):
      sentenceLstAvgwithCov = get_sentence_vec_avg_with_cov2(sentences,model)
      return sentenceLstAvgwithCov
  elif(Avg and (not SD)):
      sentencesLstAvg = get_sentence_vec_avg(sentences,model)
      return sentencesLstAvg
  elif((not Avg) and SD):
      sentencesLstSD = get_sentence_vec_SD_only(sentences,model)
      return sentencesLstSD


def encode_midi(midi):
    frame = midi_to_notes(midi)
    gram_list = extract_gram(frame)

    sorted_gram_list = sorted(set(tuple(gram_list)))
    note2Ch = {j: encodeChinese(i) for i, j in enumerate(sorted_gram_list)}
    Ch2note = {encodeChinese(i): j for i, j in enumerate(sorted_gram_list)}

    text = ''
    for j in gram_list:
        text += note2Ch[j]

    with open('temp.txt', 'w') as file:
        file.write(text)

    sentences = sentencePiece('temp.txt', 'm', 13000, 5000)
    sentencesLst = Word2Vec(5, sentences, True, True)

    return sentencesLst

def classify(encoded):
    pred = __model.predict(encoded)

    map = ['Frédéric Chopin', 'Franz Schubert', 'Ludwig van Beethoven',
       'Johann Sebastian Bach', 'Franz Liszt', 'Sergei Rachmaninoff',
       'Robert Schumann', 'Claude Debussy', 'Joseph Haydn',
       'Wolfgang Amadeus Mozart']

    return map[pred]