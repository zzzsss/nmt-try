#!/usr/bin/env python3

# prepare dependencies for CTB with Stanford Dependencies
# -> https://nlp.stanford.edu/software/dependencies_manual.pdf
# -> stanford parser 3.3.0 + CoreNLP 3.8.0
import os, sys

# in the directory of */ptb/
BR_HOME = "../PTB3/parsed/mrg/wsj/"
OUT_DIR = "./"
dir_name = os.path.dirname(os.path.abspath(__file__))
DEP_CONVERTER = "./stanford-parser-full-2013-11-12/stanford-parser.jar"
CONVERT_PYFILE = os.path.join(dir_name, "conll.py")

TAGGER = "./stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar"

TAGGER_PROP = """
                    arch = bidirectional5words,naacl2003unknowns
            wordFunction = edu.stanford.nlp.process.AmericanizeFunction
         closedClassTags =
 closedClassTagThreshold = 40
 curWordMinFeatureThresh = 2
                   debug = false
             debugPrefix =
            tagSeparator = newline
                encoding = UTF-8
              iterations = 100
                    lang = english
    learnClosedClassTags = false
        minFeatureThresh = 2
           openClassTags =
rareWordMinFeatureThresh = 5
          rareWordThresh = 5
                  search = qn
                    sgml = false
            sigmaSquared = 0.5
                   regL1 = 0.75
               tagInside =
                tokenize = false
        tokenizerFactory =
        tokenizerOptions =
                 verbose = false
          verboseResults = false
    veryCommonWordThresh = 250
                xmlInput =
              outputFile =
            outputFormat = tsv
     outputFormatOptions =
                nthreads = 4
"""

def printing(s):
    print(s, flush=True)

def system(cmd, pp=False, ass=True):
    if pp:
        printing("Executing cmd: %s" % cmd)
    n = os.system(cmd)
    if ass:
        assert n==0

def get_name(s, N=2):
    n_zero = N - len(s)
    assert n_zero >= 0
    for i in range(n_zero):
        s = "0" + s
    return s

def get_pairs(nway, train, dev, test):
    pairs = []
    assert len(train) % nway == 0
    for i in range(nway):
        tr, tt = [], []
        for one in train:
            if (one-train[0]) // (len(train)//nway) == i:
                tt.append(one)
            else:
                tr.append(one)
        pairs.append((tr, tt))
    pairs.append((train, dev+test))
    return pairs

def main(from_step=0):
    train = [i for i in range(2, 21+1)]
    dev = [22]
    test = [23]
    printing("From step %s" % from_step)
    if from_step<=0:
        printing("Step0: convert 2-23 files")
        for i in train+dev+test:
            nn = get_name(str(i))
            FILE_BASE = OUT_DIR+nn
            system("cat %s/%s/*.mrg >%s" % (BR_HOME, nn, FILE_BASE+".ptb"), True)
            system("java -cp %s -mx1g edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile %s -conllx -basic > %s" % (DEP_CONVERTER, FILE_BASE+".ptb", FILE_BASE+".conll"), True)
            system("python3 %s %s conll06 %s pos" % (CONVERT_PYFILE, FILE_BASE+".conll", FILE_BASE+".pos"), True)
    if from_step<=1:
        printing("Step1: tagging them")
        paris = get_pairs(10, train, dev, test)
        PROP = OUT_DIR+"props"
        with open(PROP, "w") as fd:
            fd.write(TAGGER_PROP)
        for tr, tt in paris:
            printing("Training on %s, tagging on %s." % (tr, tt))
            trn, ttn = [get_name(str(i)) for i in tr], [get_name(str(i)) for i in tt]
            # concat training files
            NAME_BASE = OUT_DIR+"TR%s-TT%s_%s" % (len(trn), len(ttn), "".join(ttn))
            TRAIN_NAME = NAME_BASE+".pos"
            MODEL_NAME = NAME_BASE+".model"
            system("cat %s > %s" % (" ".join([OUT_DIR+z+".pos" for z in trn]), TRAIN_NAME), True)
            # train
            system('java -cp %s edu.stanford.nlp.tagger.maxent.MaxentTagger -prop %s -trainFile "format=TSV,wordColumn=0,tagColumn=1,%s" -model %s' % (TAGGER, PROP, TRAIN_NAME, MODEL_NAME), True)
            # test / tag
            for one in ttn:
                TEST_NAME = OUT_DIR+one+".pos"
                OUT_NAME = OUT_DIR+one+".tag"
                system('java -cp %s edu.stanford.nlp.tagger.maxent.MaxentTagger -testFile "format=TSV,wordColumn=0,tagColumn=1,%s" -model %s |& grep -E "^Total"' % (TAGGER, TEST_NAME, MODEL_NAME), True)
                system('java -cp %s edu.stanford.nlp.tagger.maxent.MaxentTagger -textFile "format=TSV,wordColumn=0,tagColumn=1,%s" -model %s -tokenize false -outputFormat tsv > %s' % (TAGGER, TEST_NAME, MODEL_NAME, OUT_NAME), True)
    if from_step <= 2:
        printing("Step2: final step")
        for name, ll in (["train", train], ["dev", dev], ["test", test]):
            system("cat %s > %s" % (" ".join([OUT_DIR+get_name(str(z))+".conll" for z in ll]), name+".gold"), True)
            system("cat %s > %s" % (" ".join([OUT_DIR+get_name(str(z))+".tag" for z in ll]), name+".tag"), True)
            system("python3 %s %s.gold conll06 %s.auto conll06 %s.tag pos" % (CONVERT_PYFILE, name, name, name), True)

if __name__ == '__main__':
    from_step = 0
    try:
        x = int(sys.argv[1])
        from_step = x
    except:
        pass
    main(from_step)
