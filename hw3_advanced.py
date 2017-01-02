import hw3_corpus_tool as hw3
import pycrfsuite as pycrf
import os
import sys
import operator
import functools
import string

#input_dir = sys.argv[1]
#test_dir = sys.argv[2]
#output_file = sys.argv[3]

input_dir = "cl"
test_dir = "te"
output_file = "nin_output.txt"


#output_file = sys.argv[2]
#fout = open(output_file,'w', encoding="latin1")
fout = open(output_file,'w')


def makeseq(utter_perfile):
    punctuation = string.punctuation
    digit = string.digits
    letter = string.ascii_letters
    x_seq = []
    y_seq = []
    firstutter = 0
    speakerinit = 0
    tot = len(utter_perfile)
    count = 0
    for act_tag, speaker, pos, text in utter_perfile:
        x_sub_train = []
        y_seq.append(act_tag)
        # feature 1
        if speakerinit == 0:
            prev_speaker = speaker
            speakerinit = 1
            x_sub_train.append(str(0))
        else:
            curr_speaker = speaker
            if prev_speaker == curr_speaker:
                x_sub_train.append(str(0))
            else:
                x_sub_train.append(str(1))
                prev_speaker = curr_speaker
        # feature 2
        if firstutter == 0:
            x_sub_train.append(str(0))
            firstutter = 1
        else:
            x_sub_train.append(str(1))
        # feature 3 & 4
        feature3 = []
        feature4 = []
        # advanced feature
        feature5 = []
        if pos:
            for p in range(0,len(pos)):
                feature3.append("TOKEN_" + pos[p].token)
                feature4.append("POS_" + pos[p].pos)
                if p < (len(pos)-1):
                    feature5.append("BIGRAM_"+pos[p].token+"_"+pos[p+1].token)
                else:
                    feature5.append("BIGRAM_"+pos[p].token)
            for f3 in feature3:
                x_sub_train.append(f3)
            for f4 in feature4:
                x_sub_train.append(f4)
            for f5 in feature5:
                x_sub_train.append(f5)
        #advanced feature
        words = text.split()
        x_sub_train.append(str(len(words)))

        numbers = sum((c=='?') for c in text)
        if numbers > 0:
            x_sub_train.append(str(1))
        else:
                x_sub_train.append(str(0))

        letter_count = filter(functools.partial(operator.contains, letter), text)
        #x_sub_train.append(str(letter_count))

        count += 1
        if count == tot:
            x_sub_train.append(str(1))
        else:
            x_sub_train.append(str(0))
        x_seq.append(x_sub_train)

    return x_seq, y_seq




x_seq_train = []
y_seq_train = []
for dialogutter in hw3.get_data(input_dir):
    x_seq, y_seq = makeseq(dialogutter)
    x_seq_train += x_seq
    y_seq_train += y_seq



trainer = pycrf.Trainer(verbose=False)

trainer.append(x_seq_train, y_seq_train)


trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 200,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.params()
trainer.train('ninja.crfsuite')

################################### Testing
tagger = pycrf.Tagger()
tagger.open('ninja.crfsuite')

match = 0
mistmatch = 0


for root, dirs, files in os.walk(test_dir):
    for file in files:
        if file.endswith(".csv"):
            fullfilename = os.path.join(root, file)
            utter_perfile = hw3.get_utterances_from_filename(fullfilename)
            x_seq, y_seq = makeseq(utter_perfile)
            y_seq_op = tagger.tag(x_seq)
            fout.write("Filename=\"" + file + "\"\n")
            for k in range(0, len(y_seq)):
                fout.write(y_seq_op[k]+"\n")
                if y_seq[k] == y_seq_op[k]:
                    match += 1
                else:
                    mistmatch += 1
            fout.write("\n")


accuracy = match/(match+mistmatch)

print("Advanced Accuracy - "+str(accuracy))