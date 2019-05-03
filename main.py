import collections
from collections import Counter
from peeps_finder import *
import re
import json
import time
from textblob import TextBlob
from tqdm import tqdm

# important attributes
import_attr = ['email', 'phone', 'occupation', 'position held', 'organization',
               'educated at', 'known for', 'knows', 'country', 'keyword']

# remove these attributes
blacklist_attr = ['number', 'important date', 'important time', 'family name']

# These words specify that the user is done validating information.
# Type these instead of a number to skip the validation step
end_words = ['end', 'stop', 'done', 'exit']


def print_attr(name, values, attr_max_len=50):
    print(str(name).title())
    for i, v in enumerate(values):
        print((('[{0:2}] {1:' + str(attr_max_len) + '} {2:10}').format(i + 1, v[0].strip()[:attr_max_len],
                                                                            v[1].strip())))


def clean_info(info):
    pdata = collections.defaultdict(list)

    emails = sorted(list(Counter(info['email']).items()), key=lambda x: x[1], reverse=True)[:5]
    pdata['email'] = [(i[0], 'Medium confidence (seen {} times)'.format(i[1])) for i in emails]

    emails = sorted(list(Counter(info['phone']).items()), key=lambda x: x[1], reverse=True)[:5]
    pdata['phone'] = [(i[0], 'Medium confidence (seen {} times)'.format(i[1])) for i in emails]

    for i in info['rel_extr']:
        pdata[i[0]].append((i[1], 'High confidence'))
    for i in info['named_entities']:
        pdata[i[0]].append((i[1], 'High confidence (seen {} times)'.format(i[2]) if i[2] > 3
        else 'Medium confidence (seen {} times)'.format(i[2])))

    keywords = sorted(list(Counter(info['noun_phrases'] + info['tfidf']).items()), key=lambda x: x[1], reverse=True)
    keywords = [(i[0], 'Medium confidence (seen {} times)'.format(i[1]) if i[1] > 1
    else 'Low confidence (seen 1 times)') for i in keywords]
    pdata['keyword'] = keywords[:20]
    return pdata


def print_all_info(info):
    for attr in import_attr:
        if attr in info:
            print_attr(attr, info[attr])
    for attr in info.keys():
        if attr not in import_attr + blacklist_attr and attr is not None:
            print_attr(attr, info[attr])


def extract_nums(string_input, max_num):
    return [int(i[0].replace(',', '')) - 1 for i in re.findall(r"([\d]+(\s|\,|$)){1}", string_input)
            if int(i[0].replace(',', '')) <= max_num]


def user_search(peeps_finder, name=None, search_term=None, topn=20):
    if name is None:
        name = input("\nWho would you like to search for? ").strip()
        name_check = re.match(r"([a-zA-Z]+(\s|$)){2}", name)
        if name_check is None or name_check.group() != name:
            print("I'm sorry, I didn't get that. Please enter a name consisting of two words separated by a space")
            return user_search(peeps_finder)

    print("\nSearching for {}. Please wait ...".format(name if search_term is None else search_term))
    info = peeps_finder.retrieve_person_data(name, search=search_term, topn=topn)
    info = clean_info(info)
    print("Found some information\n")
    return info, name


def user_validation(info):
    print("Please validate the following information. Type 'done' when done.")
    attrs_to_ask = []
    for attr in import_attr:
        if attr in info:
            attrs_to_ask.append(attr)
    for attr in info.keys():
        if attr not in import_attr + blacklist_attr and attr is not None:
            attrs_to_ask.append(attr)

    keep = []
    for attr in attrs_to_ask:
        if len(info[attr]) == 0:
            continue

        print_attr(attr, info[attr])
        num_input = input('\nWhat number(s) would you like to keep? ')
        if num_input.lower().strip() in end_words: break
        nums = extract_nums(num_input, len(info[attr]))

        if len(nums) > 0 and nums[0] != -1:
            combined_values = ", ".join([str(info[attr][n][0])[:50] for n in nums])
            print("\t{}: {}".format(attr, combined_values))
            keep.append((attr, combined_values))
        else:
            print('\tNot keeping any {} values'.format(attr))
        print("")
    print("Validation of collected information is complete!\n")
    print("I am recording the following data:")
    for i in keep:
        print("  {:20}-     {:70}".format(i[0], i[1]))
    print()
    return keep


def user_get_feedback(name, keep):
    feedback = input("How do you rate the collected data (great, ok, bad, etc)? ")
    sentiment = TextBlob(feedback).sentiment.polarity
    if sentiment < 0.5:
        if input("Would you like to make a better search? ").lower().strip() in ['yes', 'sure', 'ok', 'yep', 'y']:
            print("Please select a new search term or provide your own")
            for i, a in enumerate(keep):
                print("  [{:2}]: {} {}".format(i + 1, name, a[1]))
            redo = input()
            nums = extract_nums(redo, len(keep))
            search_term = str(name) + ' ' + keep[nums[0]][1] if len(nums) > 0 else str(redo)
            print("Redoing search with the phrase {}\n".format(search_term))
            return search_term, sentiment, feedback
    return False, sentiment, feedback


def run_session(peeps_finder):
    keep = None
    feedbacks = []

    keep_going = True
    search_term = None
    name = None
    while keep_going:
        info, name = user_search(peeps_finder, name=name, search_term=search_term)
        keep = user_validation(info)
        search_term, sentiment, feedback = user_get_feedback(name, keep)
        feedbacks.append((feedback, sentiment, str(keep), name))
        if not search_term:
            keep_going = False

    keep.insert(0, ('name', name))

    fileoutput = "./{}_data.json".format(name.lower().replace(" ", "_"))
    print("Wrote the following information to", fileoutput)
    print(keep)
    with open(fileoutput, 'w') as f:
        json.dump(keep, f, indent=2)

    with open('logfile.json', 'a') as f:
        json.dump(feedbacks, f)


if __name__ == "__main__":
    peeps_finder = PeepsFinder()
    run_session(peeps_finder)

