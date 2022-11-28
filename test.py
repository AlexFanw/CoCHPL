import pickle
if __name__ == "__main__":
    # f = open('dict_word.pkl', 'rb')
    # for line in f:
    #     print(line)
    word = pickle.load(open("fea_large.pkl", 'rb'), encoding='utf-8')
    print("")

