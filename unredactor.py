
import sklearn
import spacy
import glob
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score

sample = ["""Rare is the red carpet where a single look summarizes the entire event. But gaze upon Lakeith Stanfield, with freshly auburn hair, in a Saint Laurent jumpsuit with a plunging v-neck and a pointy white collared shirt. It was super ’70s, with Stanfield’s belted waist and broad shoulders and dagger-point collar. But it was also titillatingly fluid. Kinda racy. Very sexy. He looked hot, weird, and charismatic. Up for Best Supporting Actor for his work in Judas and the Black Messiah, he looked like an artist. He also just looked awesome. It was like a treatise on the greater state of men’s style: this is what’s going on here, now.""",
        """If past genderfluid styles have been gauntlet-throwing statements of glamour, like Billy Porter’s velvet tuxedo gown in 2019, Stanfield’s was subtler, which is a nice reading of the room, but it was also a more provocative use of fashion. “I wanted to express who he is as a person: someone who is equally thoughtful as he is playful,” his stylist, Julie Ragolia, explained in a text message. Saint Laurent designer Anthony Vaccarello’s Spring 2021 women’s collection “stayed with me,” she said, and they decided to adapt a piece from it, a lean jumpsuit that recalls the eponymous designer’s fondness for safari jackets, for Stanfield: “In thinking of a way to balance the formality of such a show, this special nomination for LaKeith, and the seriousness of the times we are all living in, coming to such a look just felt thoughtful, while still being celebratory.” Ragolia also noted that the look was made with sustainable materials.""",
        """If there’s one look from the Oscars red carpet last night that got people talking, it was actor LaKeith Stanfield’s gender-bending, perfectly fitting, super-sexy Saint Laurent jumpsuit. He may not have won the award for Best Supporting Actor for his role in Judas and the Black Messiah, but he took home the prize for Best Dressed in my book.""",
        """FBI informant William O’Neal (LaKeith Stanfield) infiltrates the Illinois Black Panther Party and is tasked with keeping tabs on their charismatic leader, Chairman Fred Hampton (Daniel Kaluuya). A career thief, O’Neal revels in the danger of manipulating both his comrades and his handler, Special Agent Roy Mitchell. Hampton’s political prowess grows just as he’s falling in love with fellow revolutionary Deborah Johnson. Meanwhile, a battle wages for O’Neal’s soul. Will he align with the forces of good? Or subdue Hampton and The Panthers by any means, as FBI Director J. Edgar Hoover commands?"""]

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

    #print(features)

    v = DictVectorizer(sparse=False)
    train_X = v.fit_transform([x for (x,y) in features[:]])
    train_y = [y for (x,y) in features[:]]
   # print(train_X)
    #print(train_y)
    test_X = v.fit_transform([x for (x,y) in features[:]])
    test_y = [y for (x,y) in features[:]]
    print(test_X)

    #clf = DecisionTreeClassifier(criterion="entropy")
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(train_X, train_y)

    print("Decison Tree: ", clf.predict(test_X),  test_y)

    #print("Cross Val Score: ", cross_val_score(clf,
    #                                          v.fit_transform([x for (x,y) in features]),
     #                                          [y for (x,y) in features],
      #                                         cv=2))



if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    main()
