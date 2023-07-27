# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.data.datasets.register_coco import register_coco_instances
import os

categories = [{'id': 1, 'name': 'Antiques'}, {'id': 2, 'name': 'Berets'}, {'id': 3, 'name': 'chicken'},
              {'id': 4, 'name': 'CPU'}, {'id': 5, 'name': 'cup'}, {'id': 6, 'name': 'eyebrow pencil'},
              {'id': 7, 'name': 'fishing supplies'}, {'id': 8, 'name': "men's wallet"}, {'id': 9, 'name': 'pork'},
              {'id': 10, 'name': 'router'}, {'id': 11, 'name': 'Bodhi'}, {'id': 12, 'name': 'Coffee machine'},
              {'id': 13, 'name': 'Desktop'}, {'id': 14, 'name': 'Guitar'}, {'id': 15, 'name': 'jacket pants'},
              {'id': 16, 'name': 'ocarina'}, {'id': 17, 'name': 'Scooter'}, {'id': 18, 'name': 'Shoebox'},
              {'id': 19, 'name': 'Side table'}, {'id': 20, 'name': 'backpack'}, {'id': 21, 'name': 'badminton racket'},
              {'id': 22, 'name': 'baseball cap'}, {'id': 23, 'name': 'basketball shoes'},
              {'id': 24, 'name': 'leg socks'}, {'id': 25, 'name': 'Camera'}, {'id': 26, 'name': 'cell phone'},
              {'id': 27, 'name': "children's clothing"}, {'id': 28, 'name': 'set'}, {'id': 29, 'name': 'biscuit cake'},
              {'id': 30, 'name': 'cupboard'}, {'id': 31, 'name': 'diamond ring'}, {'id': 32, 'name': 'instant noodles'},
              {'id': 33, 'name': 'dried flowers'}, {'id': 34, 'name': 'dumbbel'}, {'id': 35, 'name': 'luxury watch'},
              {'id': 36, 'name': 'fish mouth shoes'}, {'id': 37, 'name': 'Vest'}, {'id': 38, 'name': 'glass'},
              {'id': 39, 'name': 'wine'}, {'id': 40, 'name': 'jeans'}, {'id': 41, 'name': 'kiwi fruit'},
              {'id': 42, 'name': 'Military Clothing'}, {'id': 43, 'name': 'pillow cushion'}, {'id': 44, 'name': 'pipe'},
              {'id': 45, 'name': 'platform shoes'}, {'id': 46, 'name': 'rain boots'},
              {'id': 47, 'name': 'ski equipment'}, {'id': 48, 'name': 'smart toilet'},
              {'id': 49, 'name': 'smart watch'}, {'id': 50, 'name': 'Somatosensory car'},
              {'id': 51, 'name': 'storage card'}, {'id': 52, 'name': 'telescope'}, {'id': 53, 'name': 'toilet'},
              {'id': 54, 'name': 'towel'}, {'id': 55, 'name': 'volleyball'}, {'id': 56, 'name': 'Chopsticks'},
              {'id': 57, 'name': 'Erhu'}, {'id': 58, 'name': 'Gloves'}, {'id': 59, 'name': 'One machine'},
              {'id': 60, 'name': 'Scissors'}, {'id': 61, 'name': 'Umbrella'}, {'id': 62, 'name': 'brooch'},
              {'id': 63, 'name': 'car charger'}, {'id': 64, 'name': 'casserole'}, {'id': 65, 'name': 'chandelier'},
              {'id': 66, 'name': 'cotton shoes'}, {'id': 67, 'name': 'desk lamp'}, {'id': 68, 'name': 'desk'},
              {'id': 69, 'name': 'Paper'}, {'id': 70, 'name': 'drum kit'}, {'id': 71, 'name': 'earphone'},
              {'id': 72, 'name': 'electronic scale'}, {'id': 73, 'name': 'Fax equipment'},
              {'id': 74, 'name': 'fish tank'}, {'id': 75, 'name': 'football'}, {'id': 76, 'name': 'hiking shoes'},
              {'id': 77, 'name': 'humidifier'}, {'id': 78, 'name': 'keyboard'}, {'id': 79, 'name': 'key chain'},
              {'id': 80, 'name': 'Mobile hard disk'}, {'id': 81, 'name': 'pajamas'}, {'id': 82, 'name': 'pendant'},
              {'id': 83, 'name': 'photo frame'}, {'id': 84, 'name': 'pingpong ball'}, {'id': 85, 'name': 'plum'},
              {'id': 86, 'name': 'pullover hoodie'}, {'id': 87, 'name': 'scented candles'},
              {'id': 88, 'name': 'seat cushion'}, {'id': 89, 'name': 'sneakers'}, {'id': 90, 'name': 'slippers'},
              {'id': 91, 'name': 'sports accessories'}, {'id': 92, 'name': 'sports shoes'},
              {'id': 93, 'name': 'toothbrush'}, {'id': 94, 'name': 'trolley case'}, {'id': 95, 'name': 'wallet'},
              {'id': 96, 'name': 'wheelchair'}, {'id': 97, 'name': 'wok'}, {'id': 98, 'name': 'work shoes'},
              {'id': 99, 'name': 'yoga mat'}, {'id': 100, 'name': 'Chair'}, {'id': 101, 'name': 'Chinese tunic suit'},
              {'id': 102, 'name': 'Computer Desk'}, {'id': 103, 'name': 'Hoodie'}, {'id': 104, 'name': 'led lights'},
              {'id': 105, 'name': 'Pants'}, {'id': 106, 'name': 'Shoulder Bags'}, {'id': 107, 'name': 'Storage Box'},
              {'id': 108, 'name': 'suona'}, {'id': 109, 'name': 'Temporary parking sign'},
              {'id': 110, 'name': 'blanket'}, {'id': 111, 'name': 'books'}, {'id': 112, 'name': 'skirt'},
              {'id': 113, 'name': 'cake'}, {'id': 114, 'name': 'Cardigan'}, {'id': 115, 'name': 'casual clothes'},
              {'id': 116, 'name': 'cutting board'}, {'id': 117, 'name': 'down jacket'},
              {'id': 118, 'name': 'dress shoes'}, {'id': 119, 'name': 'driving recorder'},
              {'id': 120, 'name': 'electric fan'}, {'id': 121, 'name': 'electric oven'},
              {'id': 122, 'name': 'emergency light'}, {'id': 123, 'name': 'faux leather jacket'},
              {'id': 124, 'name': 'floor'}, {'id': 125, 'name': 'flowers'}, {'id': 126, 'name': 'fur'},
              {'id': 127, 'name': 'hairy crab'}, {'id': 128, 'name': 'handbag'}, {'id': 129, 'name': 'Harmonica'},
              {'id': 130, 'name': 'laptop bag'}, {'id': 131, 'name': 'mango'}, {'id': 132, 'name': 'olives'},
              {'id': 133, 'name': 'potted plant'}, {'id': 134, 'name': 'server'}, {'id': 135, 'name': 'silver ring'},
              {'id': 136, 'name': 'skirt'}, {'id': 137, 'name': 'sports bag'}, {'id': 138, 'name': 'stockings'},
              {'id': 139, 'name': 'tent'}, {'id': 140, 'name': 'tire'}, {'id': 141, 'name': 'violin'},
              {'id': 142, 'name': 'wine cabinet'}, {'id': 143, 'name': 'woolen coat'},
              {'id': 144, 'name': 'Bracelets'}, {'id': 145, 'name': 'Central air conditioning'},
              {'id': 146, 'name': 'Martin boots'}, {'id': 147, 'name': 'man shoes'}, {'id': 148, 'name': 'Micro-wave oven'},
              {'id': 149, 'name': 'roller skateboard'}, {'id': 150, 'name': 'SLR camera'}, {'id': 151, 'name': 'tea set'}, {'id': 152, 'name': 'apple'}, {'id': 153, 'name': 'balance car'}, {'id': 154, 'name': 'Shirt'}, {'id': 155, 'name': 'bike'}, {'id': 156, 'name': 'bikini'}, {'id': 157, 'name': 'biscuit'}, {'id': 158, 'name': 'car'}, {'id': 159, 'name': 'stove'}, {'id': 160, 'name': 'Cotton clothes'}, {'id': 161, 'name': 'dining table'}, {'id': 162, 'name': 'dress'},
              {'id': 163, 'name': 'electronic organ'}, {'id': 164, 'name': 'fanny pack'}, {'id': 165, 'name': 'game console'}, {'id': 166, 'name': 'garden light'}, {'id': 167, 'name': 'graphics card'}, {'id': 168, 'name': 'hard disk'},
              {'id': 169, 'name': 'the iron'}, {'id': 170, 'name': 'leather shoes'},
              {'id': 171, 'name': 'motherboard'}, {'id': 172, 'name': 'motorcycle helmet'}, {'id': 173, 'name': 'passion fruit'}, {'id': 174, 'name': 'phone case'}, {'id': 175, 'name': 'pitaya'}, {'id': 176, 'name': "plus size women's clothing"}, {'id': 177, 'name': 'pool table'}, {'id': 178, 'name': 'sandals'}, {'id': 179, 'name': 'ski suit'}, {'id': 180, 'name': 'snow boots'}, {'id': 181, 'name': 'sock'}, {'id': 182, 'name': 'sofa'}, {'id': 183, 'name': 'sphygmomanometer'}, {'id': 184, 'name': 'suit'}, {'id': 185, 'name': 'sunglasses'}, {'id': 186, 'name': 'Thermal jug'}, {'id': 187, 'name': 'pantyhose'}, {'id': 188, 'name': 'hiking shoes'}, {'id': 189, 'name': 'Chassis'},
              {'id': 190, 'name': 'Safety seats'}, {'id': 191, 'name': 'Vacuum cleaner'}, {'id': 192, 'name': 'djembe'}, {'id': 193, 'name': 'bag'}, {'id': 194, 'name': 'bra set'}, {'id': 195, 'name': 'canvas shoes'}, {'id': 196, 'name': 'Carpet Mats'}, {'id': 197, 'name': 'ceramic tile'}, {'id': 198, 'name': 'cherry'}, {'id': 199, 'name': 'couple pajamas'},
              {'id': 200, 'name': 'drone'}, {'id': 201, 'name': 'Dumplings'}, {'id': 202, 'name': 'faucet'}, {'id': 203, 'name': 'floor mat'},
              {'id': 204, 'name': 'golf'}, {'id': 205, 'name': 'handcrafted gift'}, {'id': 206, 'name': 'lipstick'}, {'id': 207, 'name': 'long johns'}, {'id': 208, 'name': 'bags'}, {'id': 209, 'name': 'lute'}, {'id': 210, 'name': "men's belt"}, {'id': 211, 'name': 'mom shoes'}, {'id': 212, 'name': 'monitor'}, {'id': 213, 'name': 'notebook'}, {'id': 214, 'name': 'pea'}, {'id': 215, 'name': 'Pens'}, {'id': 216, 'name': 'quick dry clothes'}, {'id': 217, 'name': 'shelf'}, {'id': 218, 'name': 'remote control car'}, {'id': 219, 'name': 'road vehicles'}, {'id': 220, 'name': 'scarf'}, {'id': 221, 'name': 'seedlings'}, {'id': 222, 'name': 'shawl'}, {'id': 223, 'name': 'shirt'}, {'id': 224, 'name': 'small'}, {'id': 225, 'name': 'suit'},
              {'id': 226, 'name': 'Swimming Goggles'}, {'id': 227, 'name': 'swimming ring'}, {'id': 228, 'name': 'travel bag'}, {'id': 229, 'name': 'tub'}, {'id': 230, 'name': 'video camera'}, {'id': 231, 'name': 'water dispenser'}, {'id': 232, 'name': "women's boots"}, {'id': 233, 'name': "women's silk scarves"}, {'id': 234, 'name': 'engine oil'}, {'id': 235, 'name': 'Grapefruit'}, {'id': 236, 'name': 'guzheng'}, {'id': 237, 'name': 'treadmill'}, {'id': 238, 'name': 'Wardrobe'}, {'id': 239, 'name': 'Wedding dress'}, {'id': 240, 'name': 'apricot'}, {'id': 241, 'name': 'baby cart'}, {'id': 242, 'name': 'sandals'}, {'id': 243, 'name': 'floor'}, {'id': 244, 'name': 'ceiling lamp'}, {'id': 245, 'name': 'Bracelet'}, {'id': 246, 'name': 'cigarette case'}, {'id': 247, 'name': 'cigarette holder'}, {'id': 248, 'name': 'coat'}, {'id': 249, 'name': 'drinks'}, {'id': 250, 'name': 'electric kettle'}, {'id': 251, 'name': 'electronic drum'}, {'id': 252, 'name': 'fire extinguisher'}, {'id': 253, 'name': 'fleece pants'}, {'id': 254, 'name': 'dress'}, {'id': 255, 'name': 'Green potted plants'}, {'id': 256, 'name': 'lens'},
              {'id': 257, 'name': 'messenger bag'}, {'id': 258, 'name': 'microphone'}, {'id': 259, 'name': 'mountain bike'}, {'id': 260, 'name': 'mouse'}, {'id': 261, 'name': 'outdoor accessories'},
              {'id': 262, 'name': 'coat'}, {'id': 263, 'name': 'pan'}, {'id': 264, 'name': 'piano'}, {'id': 265, 'name': 'ravioli'}, {'id': 266, 'name': 'running shoes'}, {'id': 267, 'name': 'water tank'}, {'id': 268, 'name': 'soccer shoes'}, {'id': 269, 'name': 'Sweatpants'}, {'id': 270, 'name': 'stool'}, {'id': 271, 'name': 'sweet dumpling'}, {'id': 272, 'name': 'switch socket'}, {'id': 273, 'name': 'tablet'}, {'id': 274, 'name': 'training shoes'}, {'id': 275, 'name': 'Jacket'}, {'id': 276, 'name': 'Breaker'}, {'id': 277, 'name': 'whisky'}, {'id': 278, 'name': 'windbreaker'}, {'id': 279, 'name': 'Casual snacks'}, {'id': 280, 'name': 'Chiffon shirt'},
              {'id': 281, 'name': 'Electric pressure cooker'}, {'id': 282, 'name': 'High heel'}, {'id': 283, 'name': 'mahjong'},
              {'id': 284, 'name': 'Sweeping robot'}, {'id': 285, 'name': 'TV cabinet'}, {'id': 286, 'name': 'Water Purifier'}, {'id': 287, 'name': 'Zongzi'}, {'id': 288, 'name': 'sports camera'}, {'id': 289, 'name': 'air purifier'}, {'id': 290, 'name': 'bedside table'}, {'id': 291, 'name': 'bookshelf'}, {'id': 292, 'name': 'bow tie'}, {'id': 293, 'name': 'bra'}, {'id': 294, 'name': 'bracelet'}, {'id': 295, 'name': 'candy'}, {'id': 296, 'name': 'carpet'}, {'id': 297, 'name': 'cloth shoes'}, {'id': 298, 'name': 'dishwasher'}, {'id': 299, 'name': 'downlight'}, {'id': 300, 'name': 'durian'}, {'id': 301, 'name': 'earring'}, {'id': 302, 'name': 'flute'}, {'id': 303, 'name': 'hair dryer'}, {'id': 304, 'name': 'boots'}, {'id': 305, 'name': 'lighter'}, {'id': 306, 'name': "men's underwear"}, {'id': 307, 'name': 'office cabinet'}, {'id': 308, 'name': 'peaked cap'}, {'id': 309, 'name': 'polo shirt'}, {'id': 310, 'name': 'rain boots'}, {'id': 311, 'name': 'rice wine'}, {'id': 312, 'name': 'shoes'}, {'id': 313, 'name': 'short jacket'}, {'id': 314, 'name': 'small suit'}, {'id': 315, 'name': 'socket'}, {'id': 316, 'name': 'speakers'},
              {'id': 317, 'name': 'succulent plants'}, {'id': 318, 'name': 'thermal underwear'}, {'id': 319, 'name': 'traditional cloth shoes'}, {'id': 320, 'name': 'tripod'}, {'id': 321, 'name': 'window screening'}, {'id': 322, 'name': "women's underwear"}, {'id': 323, 'name': 'workstation'}, {'id': 324, 'name': 'Bluetooth earphone'}, {'id': 325, 'name': 'Chestnut'}, {'id': 326, 'name': 'Cycling Equipment'}, {'id': 327, 'name': 'Decorative lights'}, {'id': 328, 'name': 'Face mask'}, {'id': 329, 'name': 'height increasing shoes'}, {'id': 330, 'name': 'cycling outfit'}, {'id': 331, 'name': 'Liquor'},
              {'id': 332, 'name': 'Mobile Phone Cases'}, {'id': 333, 'name': 'peach'}, {'id': 334, 'name': 'anklet'}, {'id': 335, 'name': 'badminton'}, {'id': 336, 'name': 'bathroom supplies'}, {'id': 337, 'name': 'beer'}, {'id': 338, 'name': 'belt'}, {'id': 339, 'name': 'billiards'}, {'id': 340, 'name': 'bowl'}, {'id': 341, 'name': 'casual shoes'}, {'id': 342, 'name': 'cherries'}, {'id': 343, 'name': 'chocolate'}, {'id': 344, 'name': 'diesel oil'}, {'id': 345, 'name': 'dress pants'}, {'id': 346, 'name': 'electric toothbrush'}, {'id': 347, 'name': 'fishing rod'}, {'id': 348, 'name': 'genuine leather coat'}, {'id': 349, 'name': 'green plants'}, {'id': 350, 'name': 'hammock'}, {'id': 351, 'name': 'headband'}, {'id': 352, 'name': 'jacket'}, {'id': 353, 'name': 'long johns'}, {'id': 354, 'name': 'massage chair'}, {'id': 355, 'name': "men's handbag"}, {'id': 356, 'name': 'harmonica'}, {'id': 357, 'name': 'printer'}, {'id': 358, 'name': 'clarinet'}, {'id': 359, 'name': 'shorts'},
              {'id': 360, 'name': 'sun visor'}, {'id': 361, 'name': 'suspenders'}, {'id': 362, 'name': 'sweater'},
              {'id': 363, 'name': 'teapot'}, {'id': 364, 'name': 'thermos cup'}, {'id': 365, 'name': 'umbrella rain gear'}, {'id': 366, 'name': 'underwear'}, {'id': 367, 'name': 'wall lamp'}, {'id': 368, 'name': 'water heater'}, {'id': 369, 'name': 'Juicer'}, {'id': 370, 'name': 'Polaroid'}, {'id': 371, 'name': 'Quilt'}, {'id': 372, 'name': 'Tooling'}, {'id': 373, 'name': 'Tracksuit'}, {'id': 374, 'name': 'Ukulele'}, {'id': 375, 'name': 'Wedges'}, {'id': 376, 'name': 'Wool cap'}, {'id': 377, 'name': 'aquarium'}, {'id': 378, 'name': 'autumn clothes'}, {'id': 379, 'name': 'bath tub'}, {'id': 380, 'name': 'beef'}, {'id': 381, 'name': 'issue card'}, {'id': 382, 'name': 'cashmere sweater'}, {'id': 383, 'name': 'casual cotton socks'}, {'id': 384, 'name': 'chest of drawers'}, {'id': 385, 'name': "children's shoes"}, {'id': 386, 'name': 'curtain'}, {'id': 387, 'name': 'decorative calligraphy and painting'}, {'id': 388, 'name': 'decorative glasses'}, {'id': 389, 'name': 'flat screen tv'}, {'id': 390, 'name': 'foot bath tray'}, {'id': 391, 'name': 'workout clothes'}, {'id': 392, 'name': 'ice cream'},
              {'id': 393, 'name': 'induction cooker'}, {'id': 394, 'name': 'leggings'},
              {'id': 395, 'name': 'Lily brand food'}, {'id': 396, 'name': "men's swimsuit"}, {'id': 397, 'name': 'milk'}, {'id': 398, 'name': 'mutton'}, {'id': 399, 'name': 'necklace'}, {'id': 400, 'name': 'parasol'}, {'id': 401, 'name': 'peanut'}, {'id': 402, 'name': 'razor'}, {'id': 403, 'name': 'saxophone'}, {'id': 404, 'name': 'shellfish'}, {'id': 405, 'name': 'swimming cap'}, {'id': 406, 'name': 'swimsuit'}, {'id': 407, 'name': 'tennis'}, {'id': 408, 'name': 'tie'}, {'id': 409, 'name': 'toothpaste'}, {'id': 410, 'name': 'trousers'}, {'id': 411, 'name': 'Vest'}, {'id': 412, 'name': 'washing machine'}, {'id': 413, 'name': "women's belt"}, {'id': 414, 'name': 'Ankle boots'}, {'id': 415, 'name': 'Building Blocks'},
              {'id': 416, 'name': 'business casual shoes'}, {'id': 417, 'name': 'Chess Mahjong'}, {'id': 418, 'name': 'Jade Bracelet'}, {'id': 419, 'name': 'guqin'}, {'id': 420, 'name': 'gourd silk'}, {'id': 421, 'name': 'Photo Wall'}, {'id': 422, 'name': 'Projector'}, {'id': 423, 'name': 'SUV'}, {'id': 424, 'name': 'Smart bracelet'}, {'id': 425, 'name': 'T-shirt'}, {'id': 426, 'name': 'Tang suit'}, {'id': 427, 'name': 'U disk'}, {'id': 428, 'name': 'air conditioner'}, {'id': 429, 'name': 'backpacks'}, {'id': 430, 'name': 'basketball'}, {'id': 431, 'name': 'boots'}, {'id': 432, 'name': 'casual pants'}, {'id': 433, 'name': 'cheongsam'}, {'id': 434, 'name': "children's bed"}, {'id': 435, 'name': 'coffee table'}, {'id': 436, 'name': 'computer chair'}, {'id': 437, 'name': 'digital camera'}, {'id': 438, 'name': 'Disinfection cabinet'},
              {'id': 439, 'name': 'drying rack'}, {'id': 440, 'name': 'electric car'}, {'id': 441, 'name': 'embroidered shoes'}, {'id': 442, 'name': 'folding bike'}, {'id': 443, 'name': 'Wine'}, {'id': 444, 'name': 'frying pan'}, {'id': 445, 'name': 'sweater'}, {'id': 446, 'name': 'maternity clothes'}, {'id': 447, 'name': 'mirrorless camera'}, {'id': 448, 'name': 'motorcycle'}, {'id': 449, 'name': 'mouse pad'}, {'id': 450, 'name': 'orange'}, {'id': 451, 'name': 'outdoor windbreaker'}, {'id': 452, 'name': 'leggings'}, {'id': 453, 'name': 'Parent-child outfit'}, {'id': 454, 'name': 'range hood'}, {'id': 455, 'name': 'refrigerator'}, {'id': 456, 'name': 'ring'}, {'id': 457, 'name': 'sea cucumber'}, {'id': 458, 'name': 'sleeping bag'}, {'id': 459, 'name': 'sports vest'}, {'id': 460, 'name': 'spotlight'}, {'id': 461, 'name': 'steamer'}, {'id': 462, 'name': 'underwear'}, {'id': 463, 'name': 'top hat'}, {'id': 464, 'name': 'trolley bag'}, {'id': 465, 'name': 'walkie talkie'}, {'id': 466, 'name': 'ladies swimsuit'}]

def _get_builtin_metadata():
    id_to_name = {x['id']: x['name'] for x in categories}
    thing_dataset_id_to_contiguous_id = {
        x['id']: i for i, x in enumerate(
            sorted(categories, key=lambda x: x['id']))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}


_PREDEFINED_SPLITS_OBJECTS365 = {
    "ovd360_train": ("ovd360/train", "ovd360/train_eng.json"),
    "ovd360_train_mosaic": ("ovd360/train", "ovd360/train_eng_mosaic.json"),
    "ovd360_train_pseudo_labels": ("ovd360/train", "ovd360/train_eng_pseudo_labels.json"),
    "ovd360_test": ("ovd360/test", "ovd360/test_eng.json"),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_OBJECTS365.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )