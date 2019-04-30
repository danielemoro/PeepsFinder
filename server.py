# IMPORTANT: Change these file paths to be in the same repository as the webs server running Peeps.
# You can find the Peeps web server repository here: https://github.com/danielemoro/peeps/tree/peeps_finder
input_file = "D:/Google Drive/BSU/BSU 2018 Fall/CS401/website/peeps_finder_in.txt"
output_file = "D:/Google Drive/BSU/BSU 2018 Fall/CS401/website/peeps_finder_out.txt"

import collections
from collections import Counter
from peeps_finder import *
import re
import json
import time
from textblob import TextBlob

# important attributes
import_attr = ['email', 'phone', 'occupation', 'position held', 'organization',
               'educated at', 'known for', 'knows', 'country', 'keyword']

# remove these attributes
blacklist_attr = ['number', 'important date', 'important time', 'family name']  # remove these attributes
end_words = ['end', 'stop', 'done', 'exit']


def user_print(string=''):
    print(string + "\n")
    if string == '':
        return
    with open(output_file, 'a') as f:
        f.write(string + "\n</br>")


def print_attr(name, values, attr_max_len=50):
    user_print(str(name).title())
    for i, v in enumerate(values):
        user_print((('[{0:2}] {1:' + str(attr_max_len) + '} {2:10}').format(i + 1, v[0].strip()[:attr_max_len],
                                                                            v[1].strip())).replace(" ", "&nbsp;"))


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


def user_print(string=''):
    print(string + "\n")
    if string == '': return
    with open(output_file, 'a') as f:
        f.write(string + "\n</br>")


def user_received_output():
    done = False
    while not done:
        with open(output_file, 'r') as f:
            lines = f.readlines()
        if len(lines) == 0:
            done = True
        else:
            time.sleep(0.01)


def user_input(string=''):
    user_print(string)
    done = False
    while not done:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        curr_input = lines[-1].strip() if len(lines) >= 1 else ''
        global last_len
        if len(lines) > last_len:
            done = True
            last_len = len(lines)
        else:
            time.sleep(0.1)
    return curr_input


def extract_nums(string_input, max_num):
    return [int(i[0].replace(',', '')) - 1 for i in re.findall(r"([\d]+(\s|\,|$)){1}", string_input)
            if int(i[0].replace(',', '')) <= max_num]


def user_search(peeps_finder, name=None, search_term=None, topn=20):
    if name is None:
        name = user_input("Who would you like to search for? ").strip()
        name_check = re.match(r"([a-zA-Z]+(\s|$)){2}", name)
        if name_check is None or name_check.group() != name:
            user_print("I'm sorry, I didn't get that. Please enter a name consisting of two words separated by a space")
            return user_search(peeps_finder)

    user_print("\nSearching for {} ... please wait ...".format(name if search_term is None else search_term))
    info = peeps_finder.retrieve_person_data(name, search=search_term, topn=topn)
    info = clean_info(info)
    user_print("Found some information</br>")
    return info, name


def user_validation(info):
    user_print("Please validate the following information. Type 'done' when done.<hr>")
    attrs_to_ask = []
    for attr in import_attr:
        if attr in info:
            attrs_to_ask.append(attr)
    for attr in info.keys():
        if attr not in import_attr + blacklist_attr and attr is not None:
            attrs_to_ask.append(attr)

    keep = []
    for attr in attrs_to_ask:
        print_attr("<div class=\".h3c\">" + attr + "</div>", info[attr])
        num_input = user_input('\n</br>What number(s) would you like to keep? ')
        if num_input.lower().strip() in end_words: break
        nums = extract_nums(num_input, len(info[attr]))

        if len(nums) > 0:
            combined_values = ", ".join([str(info[attr][n][0])[:50] for n in nums])
            user_print("\t{}: {}".format(attr, combined_values))
            keep.append((attr, combined_values))
        else:
            user_print('\tNot keeping any {} values'.format(attr))
        user_received_output()
    user_print("Validation of collected information is complete!\n")
    user_print("I am recording the following data:<hr>")
    for i in keep:
        user_print("  {:25}: {:100}".format(i[0], i[1]))
    user_print()
    return keep


def user_get_feedback(name, keep):
    feedback = user_input("</br>How do you rate the collected data (great, ok, bad, etc)? ")
    sentiment = TextBlob(feedback).sentiment.polarity
    if sentiment < 0.5:
        if user_input("Would you like to make a better search?").lower().strip() in ['yes', 'sure', 'ok', 'yep', 'y']:
            user_print("Please select a new search term or provide your own")
            for i, a in enumerate(keep):
                user_print("  [{:2}]: {} {}".format(i + 1, name, a[1]))
            redo = user_input()
            nums = extract_nums(redo, len(keep))
            search_term = str(name) + ' ' + keep[nums[0]][1] if len(nums) > 0 else str(redo)
            user_print("Redoing search with the phrase {}\n".format(search_term))
            return (search_term, sentiment, feedback)
    return (False, sentiment, feedback)


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

    with open(output_file, 'a') as f:
        json.dump(keep, f)
    print(keep)

    with open('logfile.json', 'a') as f:
        json.dump(feedbacks, f)

    if user_input() == 'END':
        print("SESSION ENDED")
        return


last_len = 0
peeps_finder = PeepsFinder()

# Run indefinitely, as long as the partner web server is running
while True:
    # Clear communications channels
    last_len = 0
    with open(output_file, 'w') as f:
        f.write("")
    with open(input_file, 'w') as f:
        f.write("")

    print("WAITING FOR NEW SESSION")
    if user_input().strip().lower() == 'start':
        print("STARTING SESSION")
        run_session(peeps_finder)
    else:
        print("Error: unexpected input")
