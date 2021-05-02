
import sklearn
import spacy
import glob
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score

sample = ["""This is a excellent start to the film career of Mickey Rooney. His talents here shows that a long career is ahead for him. The car and truck chase is exciting for the 1937 era. This start of the Andy Hardy series is an American treasure in my book. Spring Byington performance is excellent as usual. Please Mr Rooney or owners of the film rights, take a chance and get this produced on DVD. I think it would be a winner.
"""]
nlp = None

def make_features(sentence, ne="PERSON"):
    doc = nlp(sentence)
    D = []
    for e in doc.ents:
        if e.label_ == ne:
            d = {}
            # d["name"] = e.text # We want to predict this
            d["length"] = len(e.text)
            d["word_idx"] = e.start
            d["char_idx"] = e.start_char
            d["spaces"] = 1 if " " in e.text else 0
            # gender?
            # Number of occurences?
            D.append((d, e.text))
    return D

def main():
    # print(len(sample))
    dire = glob.glob("train_files/*.txt")
    data =[]
    for f in dire:
        temp_f = open(f,"r")
        data.append(temp_f.read())
        #print(data)

    features = []
    for s in data:
        features.extend(make_features(s))

    print(features)

    v = DictVectorizer(sparse=False)
    train_X = v.fit_transform([x for (x,y) in features[:-1]])
    train_y = [y for (x,y) in features[:-1]]

    test_X = v.fit_transform([x for (x,y) in features[-1:]])
    test_y = [y for (x,y) in features[-1:]]

    clf = DecisionTreeClassifier(criterion="entropy")
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(train_X, train_y)

    print("Decison Tree: ", clf.predict(test_X), clf.predict_proba(test_X), test_y)

    print("Cross Val Score: ", cross_val_score(clf,
                                              v.fit_transform([x for (x,y) in features]),
                                               [y for (x,y) in features],
                                               cv=2))



if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    main()
