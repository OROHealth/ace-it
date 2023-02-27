from src.utils import *

path = Path()

learn_ = load_learner(path/'export.pkl')

test_path = Path('./test')

test_images = get_image_files(test_path)


def ground_list():
    '''Collects the filenames of the ground truths'''

    ground_list = []
    for i in range(len(test_images)):
        ground = test_images[i]
        ground_list.append(str(ground))

    return ground_list


def ground_list_():
    '''Trims the filenames in the ground_list'''

    ground_list_ = []
    for i in range(len(ground_list)):
        ground_list_[i] = ground_list[i][-9:-5]

    return ground_list_


def predicted_list():
    '''Generates and collects the predictions'''

    predicted_list = []

    for i in range(len(test_images)):
        predicted = learn_.predict(test_images[i])[0]
        predicted_list.append(str(predicted))

    return predicted_list


def predicted_list_():
    '''Trims the filenames in the predicted_list'''

    predicted_list_ = []
    for i in range(len(predicted_list)):
        predicted_list_[i] = predicted_list[i][:4]
    return predicted_list_


def correct():
    '''Returns the number of correct predictions among the test dataset'''

    correct = 0
    for i in range(len(ground_list_)):
        if ground_list_[i] == predicted_list_[i]:
            correct += 1
    return correct


def perc_correct():
    '''Return the percentage of correct predictions among the test dataset'''

    percent_correct = correct / len(ground_list_)
    print('Percent of correct predictions on test set:',
          round(percent_correct * 100), '%')


list(zip(ground_list_(), predicted_list_()))

perc_correct()
