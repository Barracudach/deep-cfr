CARDS_RANKS = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "T": 8,
    "J": 9,
    "Q": 10,
    "K": 11,
    "A": 12
}

CARDS_SUITS = {
    "s": 0,
    "d": 1,
    "h": 2,
    "c": 3
}


def RedrawTableCards(raw_table_cards: str):
    raw_table_cards = raw_table_cards.replace(' ', '')

    formated_cards = ""
    cards_colors = []
    suit_mapping = {}

    # rank, suit, orig_suit
    table_card_list: list[(int, int, str)] = [
        [CARDS_RANKS[raw_table_cards[i]], CARDS_SUITS[raw_table_cards[i + 1]],
         raw_table_cards[i] + raw_table_cards[i + 1]]
        for i in range(0, len(raw_table_cards), 2)
    ]

    # flush draw check
    if table_card_list[0][1] == table_card_list[1][1] or table_card_list[1][1] == table_card_list[2][1] or \
            table_card_list[0][1] == table_card_list[2][1]:
        # one suit
        if table_card_list[0][1] == table_card_list[1][1] and table_card_list[1][1] == table_card_list[2][1]:
            cards_colors = ['s', 's', 's']
            table_card_list = sorted(table_card_list[0:3], key=lambda x: x[0], reverse=True) + table_card_list[3:]
        else:
            cards_colors = ['s', 's', 'd']
            not_equal_color_idx = 0
            if table_card_list[0][1] != table_card_list[1][1] and table_card_list[0][1] != table_card_list[2][1]:
                not_equal_color_idx = 0
            elif table_card_list[1][1] != table_card_list[0][1] and table_card_list[1][1] != table_card_list[2][1]:
                not_equal_color_idx = 1
            else:
                not_equal_color_idx = 2
            not_equal_card = table_card_list[not_equal_color_idx]
            table_card_list.pop(not_equal_color_idx)
            table_card_list = sorted(table_card_list[0:2], key=lambda x: x[0], reverse=True) + table_card_list[2:]
            table_card_list.insert(2, not_equal_card)
    else:
        cards_colors = ['s', 'd', 'h']

        # pair check
        if table_card_list[0][0] == table_card_list[1][0] or table_card_list[1][0] == table_card_list[2][0] or \
                table_card_list[0][0] == table_card_list[2][0]:
            if table_card_list[0][0] == table_card_list[1][0] or table_card_list[0][0] == table_card_list[2][0]:
                if table_card_list[0][0] != table_card_list[1][0]:
                    table_card_list[0], table_card_list[1] = table_card_list[1], table_card_list[0]
                elif table_card_list[0][0] != table_card_list[2][0]:
                    table_card_list[0], table_card_list[2] = table_card_list[2], table_card_list[0]
        else:
            table_card_list = sorted(table_card_list[0:3], key=lambda x: x[0], reverse=True) + table_card_list[3:]

    # iterate over the remaining cards
    for i in range(3, 5):
        if len(table_card_list) <= i:
            break
        _cards_colors = ['s', 'd', 'h', 'c']

        current_card = table_card_list[i]
        card_suit_found = False

        for j in range(0, i):
            current_compare_card = table_card_list[j]
            if current_card[1] == current_compare_card[1]:  # if suits are equal
                card_suit_found = True
                cards_colors.append(cards_colors[j])  # add existing suit
                break

        if not card_suit_found:
            available_suits = [suit for suit in _cards_colors if suit not in cards_colors]
            cards_colors.append(available_suits[0])  # add the next available suit

    for i in range(0, len(table_card_list)):
        str_rank = next((k for k, v in CARDS_RANKS.items() if v == table_card_list[i][0]), None)
        if str_rank is None:
            raise Exception(f"key not found [{table_card_list[i][0]}]")
        new_suit = cards_colors[i]
        suit_mapping[table_card_list[i][2][1]] = new_suit  # Map old suit to new suit
        formated_cards += str_rank + new_suit

    return formated_cards, suit_mapping


def RedrawHandCardsByTable(new_board_cards, hand_cards, suit_mapping):
    hand_cards = hand_cards.replace(' ', '')
    # Sort cards in hand by rank in descending order
    splited_cards = [(hand_cards[i], hand_cards[i + 1]) for i in range(0, len(hand_cards), 2)]
    sorted_hand = sorted(splited_cards, key=lambda x: CARDS_RANKS[x[0]], reverse=True)

    new_hand = ""
        
    # Recolor hand cards based on the mapping from the board
    for rank, suit in sorted_hand:
        if suit in suit_mapping:
            new_hand += rank + suit_mapping[suit]
        else:
            # If no mapping exists, assign the first available suit
            available_suits = [s for s in CARDS_SUITS if s not in new_hand and s not in new_board_cards]
            new_hand += rank + (available_suits[0] if available_suits else suit)

    return new_hand


def RedrawSingleHand(hand_cards):
    hand_cards = hand_cards.replace(' ', '')
    splited_cards = [[hand_cards[i], hand_cards[i + 1]] for i in range(0, len(hand_cards), 2)]
    sorted_hand = sorted(splited_cards, key=lambda x: CARDS_RANKS[x[0]], reverse=True)
    if sorted_hand[0][1]==sorted_hand[1][1]:
        sorted_hand[0][1]=sorted_hand[1][1]='s'
    else:
        sorted_hand[0][1]='s'
        sorted_hand[1][1]='d'
    return "".join(''.join(inner_array) for inner_array in sorted_hand)

if __name__ == '__main__': 
    board_cards = "4s3sJs8hAh"
    hand = "4d5d"
    single=RedrawSingleHand(hand)

    new_board, suit_mapping = RedrawTableCards(board_cards)
    new_hand = RedrawHandCardsByTable(new_board, hand, suit_mapping)

    print(f"{board_cards} {hand} -> {new_board} {new_hand}")
