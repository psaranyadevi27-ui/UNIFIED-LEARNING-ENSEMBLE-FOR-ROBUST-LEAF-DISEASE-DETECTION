import pickle

def save(name, val):
    with open('./Saved data/' + name + '.pkl', 'wb') as file:
        pickle.dump(val, file)


def load(name):
    with open('./Saved data/' + name + '.pkl', 'rb') as file:
        return pickle.load(file)