import spacy
import glob
import nltk
import argparse
#from spacy.matcher import Matcher
#from PyDictionary import PyDictionary
#dic=PyDictionary()
nlp = spacy.load('en_core_web_sm')
nlp.pipe_names
ner = nlp.get_pipe('ner')

#redact name method
def redactName(doc):
    nlp_doc = nlp(doc)
    re = []
    num = 0
    #print(nlp_doc)                
    for ent in nlp_doc:
        if ent.ent_type_ == "PERSON":
            re.append("(REDACTED)")
            num +=1
        else:
            re.append(ent.text)
    return ' '.join(re), num

def main(input_f):
    #print(redactName(doc))
    dire = glob.glob(input_f)
    #print(dire)
    for f in dire:
        temp_f = open(f,"r")
        #redact phone
        #temp,phone_redact = redactPhone(temp_f.read())
        # redact name
        temp,name_redact = redactName(temp_f.read())
        name = f.split(".")
        result = open(name[0]+".redacted","w")
        result.write(temp)
        result.write("Name" + " --- " + str(name_redact)+"\n")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,help="get in input")
    args = parser.parse_args()
    print(args.input)
    main(args.input)
