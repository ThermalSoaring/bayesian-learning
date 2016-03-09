#!/usr/bin/env python3
#
# Predict the next word of a sentence using a Bayesian Network. Learn the
# probabilities from Gutenburg Project text documents.
#
import re
import sys
import argparse
import pathlib
import operator
import pomegranate
import readline

#
# Clean up a paragraph
#
def cleanup(text):
    text = text.strip()
    # Convert dashes and line returns to spaces
    text = re.sub('(--|\n)', ' ', text)
    # Only allow case-insensitive a-z in words
    text = re.sub('[^a-zA-Z \'-]+', '', text)
    return text

#
# Find all the words and the next word from a Gutenburg Project text document
# and use these to construct a Bayesian network
#
def constructNetworkFromFiles(filenames, number):
    assert number > 0, "n > 0"

    paragraphs = []

    # Parse each of the files
    for filename in filenames:
        p = pathlib.Path(filename)
        assert p.is_file(), "Must specify valid file"

        # Get all the book data from this one file
        data = ""

        try:
            with p.open() as f:
                start = False
                end   = False

                for line in f:
                    if re.match("^\*\*\* END OF THIS PROJECT", line):
                        end = True

                    # Only use this line if it's between the start and end
                    if start and not end:
                        data += line

                    # Find the start and end of the document
                    if re.match("^\*\*\* START OF THIS PROJECT", line):
                        start = True
        except OSError:
            print("Could not open file")

        # Split by paragraphs
        word_usage = [{} for i in range(0, number)]
        paragraphs += data.split('\n\n')

    # Using the paragraphs obtained from all the files, look for word sequences
    for paragraph in paragraphs:
        if paragraph:
            # Split by words
            words = cleanup(paragraph).split(' ')

            # For each length sequence of words (i.e., if n=3, for a sequence
            # of length 1, 2, and 3)
            for num in range(1, number+1):
                # Look at a sliding window of the specified number of words
                # finding the frequency of each sequenc
                for i in range(num, len(words)):
                    s = ','.join([words[i-(num-j)] for j in range(0, num)])

                    if s in word_usage[num-1]:
                        word_usage[num-1][s] += 1
                    else:
                        word_usage[num-1][s] = 1

    # Total number to divide by to compute the probabilities
    total_freq = [sum(d.values()) for d in word_usage]

    # Construct the network
    word_distributions = []

    # First word, only one that isn't a CPD. Just take the frequency and divide
    # by the total to get probabilities.
    word_distributions.append(pomegranate.DiscreteDistribution(
        { key: value/total_freq[0] for (key, value) in word_usage[0].items()}))

    # The other words (2, 3, etc.)
    for i in range(1, number):
        word_distributions.append(pomegranate.ConditionalProbabilityTable(
            # CPT: word1, ..., this_word, prob(this_word)
            [key.split(',')+[value] for (key, value) in word_usage[i].items()],
            # depends on all words up to this word
            [word_distributions[j] for j in range(0, i)]))

    # Create all the nodes
    nodes = []

    for i in range(0, number):
        nodes.append(pomegranate.State(word_distributions[i],
            name="word"+str(i+1)))

    network = pomegranate.BayesianNetwork("Word Prediction")
    network.add_states(nodes)

    # Add edges from current word to all later words
    for i in range(0, number):
        for j in range(i+1, number):
            network.add_transition(nodes[i], nodes[j])

    network.bake()

    return network

#
# Make a prediction of the next word based on a string of words
#
def predict(network, number, s):
    words = s.split(" ")
    num = number

    # Start off with the desired number-1 of observations, but if we can't
    # predict based on that, keep decreasing number until we can or until we
    # find out we just can't predict.
    while num > 1:
        observations = {}

        # Start at the last word and look backwards for our observations. But,
        # to predict, we need the last node to be the next word, so don't
        # observe all number of words in our network, just observe n-1 words.
        for i, word in enumerate(words[-(num-1):]):
            observations["word"+str(i+1)] = word

        # Update the beliefs in the network
        try:
            if debug:
                print("Input string:", s)
                print("Observations:", observations)

            marginals = network.predict_proba(observations)

            # Which node do we want to look at? If there are no previous words,
            # look at the first one. If there was one, look at the second, etc.
            # If there was n-1 or more, look at the last one.
            prediction_node = min(len(words), num-1)

            # Look at the node we care about to read off the new probability
            # distribution
            prediction = marginals[prediction_node].parameters[0]

            # Sort the predictions in descending order
            sorted_prediction = sorted(prediction.items(),
                    key=operator.itemgetter(1), reverse=True)

            return sorted_prediction

        # If our observations "aren't possible," i.e. they didn't occur in the
        # training data then try again using one fewer word as an observation,
        # but only if we haven't already used that number of words
        except ZeroDivisionError:
            if len(words) >= number-1:
                num -= 1
            else:
                break
        except ValueError:
            if len(words) >= number-1:
                num -= 1
            else:
                break

    # Couldn't make a prediction
    return []

#
# Print the predictions
#
# Assuming input is sorted in descending order of most probable to least
# probable
#
def printPrediction(predictions):
    num = min(25, len(predictions))

    if num == 0:
        print("No prediction found")
    else:
        # Print out the top ones that are greater than zero
        for i in range(0, num):
            if predictions[i][1] > 0:
                print(" ", predictions[i], sep="")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='file', type=str, nargs='+',
            help='files to learn the word sequence probabilities from')
    parser.add_argument('-n', dest='number', type=int, default=3,
            help='the number of BN word nodes, observation will be of n-1 words\n'+
            '(note: start off small like n=3, will be slow if large)')
    parser.add_argument('-d', dest='debug', action='store_true',
            help='debugging information')
    args = parser.parse_args()

    # Make the debug flag global
    global debug
    debug = args.debug

    if not args.files:
        print("Must specify at least one filename")
        sys.exit(1)

    # Construct the network based on the input training files
    network = constructNetworkFromFiles(args.files, args.number)

    # Interactive prompt
    while True:
        try:
            s = input(">>> ")
            printPrediction(predict(network, args.number, s))
        except KeyboardInterrupt:
            print()
            break
        except EOFError:
            print()
            break
