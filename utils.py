'''
Informative printing function
@type: 0 is [INFO], 1 is [WARNING], otherwise is [ERROR]
'''
def iprint(content='', msg_type=0) -> None:
    if type(content) != 'str':
        content = str(content)
    if msg_type == 0:
        prefix = '[INFO] '
    elif msg_type == 1:
        prefix = '[WARNING] '
    else:
        prefix = '[ERROR] '
    print(prefix + content + '\n')


def fix_label_error(labels, type, slots):
    label_dict = dict([(l[0], l[1]) for l in labels]) if type else dict([(l['slots'][0][0], l['slots'][0][1]) for l in labels])

    GENERAL_TYPO = {
        # type
        'guesthouse': 'guest house',
        'guesthouses': 'guest house',
        'guest': 'guest house',
        'mutiple sports': 'multiple sports',
        'sports': 'multiple sports',
        'mutliple sports': 'multiple sports',
        'swimmingpool': 'swimming pool',
        'concerthall': 'concert hall',
        'concert': 'concert hall',
        'pool': 'swimming pool',
        'night club': 'nightclub',
        'mus': 'museum',
        'ol': 'architecture',
        'colleges': 'college',
        'coll': 'college',
        'architectural': 'architecture',
        'musuem': 'museum',
        'churches': 'church',
        # area
        'center': 'centre',
        'center of town': 'centre',
        'near city center': 'centre',
        'in the north': 'north',
        'cen': 'centre',
        'east side': 'east',
        'east area': 'east',
        'west part of town': 'west',
        'ce': 'centre',
        'town center': 'centre',
        'centre of cambridge': 'centre',
        'city center': 'centre',
        'the south': 'south',
        'scentre': 'centre',
        'town centre': 'centre',
        'in town': 'centre',
        'north part of town': 'north',
        'centre of town': 'centre',
        'cb30aq': 'none',
        # price
        'mode': 'moderate',
        'moderate -ly': 'moderate',
        'mo': 'moderate',
        # day
        'next friday': 'friday',
        'monda': 'monday',
        # parking
        'free parking': 'free',
        # internet
        'free internet': 'yes',
        # star
        '4 star': '4',
        '4 stars': '4',
        '0 star rarting': 'none',
        # others
        'y': 'yes',
        'any': 'dontcare',
        'n': 'no',
        'does not care': 'dontcare',
        'not men': 'none',
        'not': 'none',
        'not mentioned': 'none',
        '': 'none',
        'not mendtioned': 'none',
        '3 .': '3',
        'does not': 'no',
        'fun': 'none',
        'art': 'none',
    }

    for slot in slots:
        if slot in label_dict.keys():
            # general typos
            this_slot = label_dict[slot]
            if this_slot in GENERAL_TYPO.keys():
                this_slot = this_slot.replace(this_slot, GENERAL_TYPO[this_slot])

            # miss match slot and value
            if (
                slot == 'hotel-type' and
                this_slot in ['nigh', 'moderate -ly priced', 'bed and breakfast', 'centre', 'venetian', 'intern', 'a cheap -er hotel'] or
                slot == 'hotel-internet' and
                this_slot == '4' or
                slot == 'hotel-pricerange' and
                this_slot == '2' or
                slot == 'attraction-type' and
                this_slot in ['gastropub', 'la raza', 'galleria', 'gallery', 'science', 'm'] or
                'area' in slot and this_slot in ['moderate'] or
                'day' in slot and this_slot == 't'
            ):
                this_slot = 'none'
            elif slot == 'hotel-type' and this_slot in ['hotel with free parking and free wifi', '4', '3 star hotel']:
                this_slot = 'hotel'
            elif slot == 'hotel-star' and this_slot == '3 star hotel':
                this_slot = '3'
            elif 'area' in slot:
                if this_slot == 'no':
                    this_slot = 'north'
                elif this_slot == 'we':
                    this_slot = 'west'
                elif this_slot == 'cent':
                    this_slot = 'centre'
            elif 'day' in slot:
                if this_slot == 'we':
                    this_slot = 'wednesday'
                elif this_slot == 'no':
                    this_slot = 'none'
            elif 'price' in slot and this_slot == 'ch':
                this_slot = 'cheap'
            elif 'internet' in slot and this_slot == 'free':
                this_slot = 'yes'

            # some out-of-define classification slot values
            if (
                slot == 'restaurant-area' and
                this_slot in ['stansted airport', 'cambridge', 'silver street'] or
                slot == 'attraction-area' and
                this_slot in ['norwich', 'ely', 'museum', 'same area as hotel']
            ):
                this_slot = 'none'

    return label_dict
