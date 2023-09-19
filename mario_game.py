from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import cv2 as cv
import numpy as np
import string

# Other Constants (don't change these)
SCREEN_HEIGHT = 240
SCREEN_WIDTH = 256
MATCH_THRESHOLD = 0.9

################################################################################
# TEMPLATES FOR LOCATING OBJECTS

# ignore sky blue colour when matching templates
MASK_COLOUR = np.array([252, 136, 104])
# (these numbers are [BLUE, GREEN, RED] because opencv uses BGR colour format by default)

# filenames for object templates
image_files = {
    "mario": {
        "small": ["marioA.png", "marioB.png", "marioC.png", "marioD.png",
                  "marioE.png", "marioF.png", "marioG.png"],
        "tall": ["tall_marioA.png", "tall_marioB.png", "tall_marioC.png"],
        # Note: Many images are missing from tall mario, and I don't have any
        # images for fireball mario.
    },
    "enemy": {
        "goomba": ["goomba.png"],
        "koopa": ["koopaA.png", "koopaB.png"],
    },
    "block": {
        "brick_block": ["block1.png"],
        "ground_block": ["block2.png", "block3.png"],
        "stair_block": ["block4.png"],
        "question_block": ["questionA.png", "questionB.png", "questionC.png"],
        "pipe": ["pipe_upper_section.png", "pipe_lower_section.png"],
    },
    "item": {
        # Note: The template matcher is colourblind (it's using greyscale),
        # so it can't tell the difference between red and green mushrooms.
        "mushroom": ["mushroom_red.png"],
    }
}

# From Elements:
# Recognises Images on the screen and outputs the location (dictionary of category key)
def _get_template(filename):
    image = cv.imread(filename)
    assert image is not None, f"File {filename} does not exist."
    template = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = np.uint8(np.where(np.all(image == MASK_COLOUR, axis=2), 0, 1))
    num_pixels = image.shape[0] * image.shape[1]
    if num_pixels - np.sum(mask) < 10:
        mask = None  # this is important for avoiding a problem where some things match everything
    dimensions = tuple(template.shape[::-1])
    return template, mask, dimensions


def get_template(filenames):
    results = []
    for filename in filenames:
        results.append(_get_template(filename))
    return results


def get_template_and_flipped(filenames):
    results = []
    for filename in filenames:
        template, mask, dimensions = _get_template(filename)
        results.append((template, mask, dimensions))
        results.append((cv.flip(template, 1), cv.flip(mask, 1), dimensions))
    return results


include_flipped = {"mario", "enemy"}
image_folder_path = "mario_locate_objects/"

# generate all templates
templates = {}
for category in image_files:
    category_items = image_files[category]
    category_templates = {}
    for object_name in category_items:
        filenames = [f"{image_folder_path}{filename}" for filename in category_items[object_name]]
        if category in include_flipped or object_name in include_flipped:
            category_templates[object_name] = get_template_and_flipped(filenames)
        else:
            category_templates[object_name] = get_template(filenames)
    templates[category] = category_templates

################################################################################
# PRINTING THE GRID (for debug purposes)
colour_map = {
    (104, 136, 252): " ",  # sky blue colour
    (0, 0, 0): " ",  # black
    (252, 252, 252): "'",  # white / cloud colour
    (248, 56, 0): "M",  # red / mario colour
    (228, 92, 16): "%",  # brown enemy / block colour
}
unused_letters = sorted(set(string.ascii_uppercase) - set(colour_map.values()), reverse=True)
DEFAULT_LETTER = "?"


def _get_colour(colour):  # colour must be 3 ints
    colour = tuple(colour)
    if colour in colour_map:
        return colour_map[colour]

    # if we haven't seen this colour before, pick a letter to represent it
    if unused_letters:
        letter = unused_letters.pop()
        colour_map[colour] = letter
        return letter
    else:
        return DEFAULT_LETTER


def print_grid(obs, object_locations):
    pixels = {}
    # build the outlines of located objects
    for category in object_locations:
        for location, dimensions, object_name in object_locations[category]:
            x, y = location
            width, height = dimensions
            name_str = object_name.replace("_", "-") + "-"
            for i in range(width):
                pixels[(x + i, y)] = name_str[i % len(name_str)]
                pixels[(x + i, y + height - 1)] = name_str[(i + height - 1) % len(name_str)]
            for i in range(1, height - 1):
                pixels[(x, y + i)] = name_str[i % len(name_str)]
                pixels[(x + width - 1, y + i)] = name_str[(i + width - 1) % len(name_str)]

    # print the screen to terminal
    print("-" * SCREEN_WIDTH)
    for y in range(SCREEN_HEIGHT):
        line = []
        for x in range(SCREEN_WIDTH):
            coords = (x, y)
            if coords in pixels:
                # this pixel is part of an outline of an object,
                # so use that instead of the normal colour symbol
                colour = pixels[coords]
            else:
                # get the colour symbol for this colour
                colour = _get_colour(obs[y][x])
            line.append(colour)
        print("".join(line))


################################################################################
# LOCATING OBJECTS

def _locate_object(screen, templates, stop_early=False, threshold=MATCH_THRESHOLD):
    locations = {}
    for template, mask, dimensions in templates:
        results = cv.matchTemplate(screen, template, cv.TM_CCOEFF_NORMED, mask=mask)
        locs = np.where(results >= threshold)
        for y, x in zip(*locs):
            locations[(x, y)] = dimensions

        # stop early if you found mario (don't need to look for other animation frames of mario)
        if stop_early and locations:
            break

    #      [((x,y), (width,height))]
    return [(loc, locations[loc]) for loc in locations]


def _locate_pipe(screen, threshold=MATCH_THRESHOLD):
    upper_template, upper_mask, upper_dimensions = templates["block"]["pipe"][0]
    lower_template, lower_mask, lower_dimensions = templates["block"]["pipe"][1]

    # find the upper part of the pipe
    upper_results = cv.matchTemplate(screen, upper_template, cv.TM_CCOEFF_NORMED, mask=upper_mask)
    upper_locs = list(zip(*np.where(upper_results >= threshold)))

    # stop early if there are no pipes
    if not upper_locs:
        return []

    # find the lower part of the pipe
    lower_results = cv.matchTemplate(screen, lower_template, cv.TM_CCOEFF_NORMED, mask=lower_mask)
    lower_locs = set(zip(*np.where(lower_results >= threshold)))

    # put the pieces together
    upper_width, upper_height = upper_dimensions
    lower_width, lower_height = lower_dimensions
    locations = []
    for y, x in upper_locs:
        for h in range(upper_height, SCREEN_HEIGHT, lower_height):
            if (y + h, x + 2) not in lower_locs:
                locations.append(((x, y), (upper_width, h)))
                break
    return locations


def locate_objects(screen, mario_status):
    # convert to greyscale
    screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)

    # Initialize a nested dictionary for storing the object locations
    object_locations = {category: {} for category in templates.keys()}

    for category in templates:
        category_templates = templates[category]
        stop_early = False
        for object_name in category_templates:
            # use mario_status to determine which type of mario to look for
            if category == "mario":
                if object_name != mario_status:
                    continue
                else:
                    stop_early = True
            # pipe has special logic, so skip it for now
            if object_name == "pipe":
                continue

            # find locations of objects
            results = _locate_object(screen, category_templates[object_name], stop_early)
            # If results exist, add them to the dictionary
            if results:
                object_locations[category][object_name] = [(loc, dim) for loc, dim in results]

    # Special case for locating pipes
    pipe_locations = _locate_pipe(screen)
    if pipe_locations:
        object_locations["block"]["pipe"] = [(loc, dim) for loc, dim in pipe_locations]

    return object_locations


################################################################################
# GETTING INFORMATION AND CHOOSING AN ACTION
class Mario:
    def __init__(self, x, y, width, height, status):
        self.left_x = x
        self.top_y = y
        self.width = width
        self.height = height
        self.right_x = x + width
        self.bottom_y = y + height
        self.status = status

    def is_on_ground(self):
        return 208 <= self.bottom_y <= 209


class Enemy:
    def __init__(self, name, x, y, width, height):
        self.name = name
        self.left_x = x
        self.top_y = y
        self.width = width
        self.height = height
        self.right_x = x + width
        self.bottom_y = y + height

    def is_on_ground(self):
        return 208 <= self.bottom_y <= 209


class Ground_Block:
    def __init__(self, x, y, width, height):
        self.left_x = x
        self.top_y = y
        self.width = width
        self.height = height
        self.right_x = x + width
        self.bottom_y = y + height


class Pitfall:
    def __init__(self, x, width):
        self.left_x = x
        self.width = width


class Question_Block:
    def __init__(self, x, y, width, height):
        self.left_x = x
        self.top_y = y
        self.width = width
        self.height = height
        self.right_x = x + width
        self.bottom_y = y + height


class Pipe:
    def __init__(self, x, y, width, height):
        self.left_x = x
        self.top_y = y
        self.width = width
        self.height = height
        self.right_x = x + width
        self.bottom_y = y + height

    def is_on_ground(self):
        return 208 <= self.bottom_y <= 209


class Mushroom:
    def __init__(self, x, y, width, height):
        self.left_x = x
        self.top_y = y
        self.width = width
        self.height = height
        self.right_x = x + width
        self.bottom_y = y + height

class Stair_Block:
    def __init__(self, x, y, width, height):
        self.left_x = x
        self.top_y = y
        self.width = width
        self.height = height
        self.right_x = x + width
        self.bottom_y = y + height

class Brick_Block:
    def __init__(self, x, y, width, height):
        self.left_x = x
        self.top_y = y
        self.width = width
        self.height = height
        self.right_x = x + width
        self.bottom_y = y + height


high_jump_counter = 0


def decide_action(screen, info_dict):
    global high_jump_counter

    if high_jump_counter > 0:
        high_jump_counter -= 1
        return RIGHT_A

    ################ INITIALISING DATA STRUCTURES ################
    mario_status = info_dict["status"]
    object_locations = locate_objects(screen, mario_status)

    # Initialise mario instances
    marios_list = []
    if object_locations.get("mario"):
        if object_locations.get("mario").get(mario_status):
            for mario in object_locations.get("mario").get(mario_status):
                marios_list.append(Mario(mario[0][0], mario[0][1], mario[1][0], mario[1][1], mario_status))

    # Initialise enemy instances
    enemies_list = []
    if object_locations.get("enemy"):
        if object_locations.get("enemy").get("goomba"):
            for goomba in object_locations.get("enemy").get("goomba"):
                enemies_list.append(Enemy("goomba", goomba[0][0], goomba[0][1], goomba[1][0], goomba[1][1]))
        if object_locations.get("enemy").get("koopa"):
            for koopa in object_locations.get("enemy").get("koopa"):
                enemies_list.append(Enemy("koopa", koopa[0][0], koopa[0][1], koopa[1][0], koopa[1][1]))

    # Initialise ground_block instances
    ground_blocks_list = []
    if object_locations.get("block"):
        if object_locations.get("block").get("ground_block"):
            for ground_block in object_locations.get("block").get("ground_block"):
                ground_blocks_list.append(
                    Ground_Block(ground_block[0][0], ground_block[0][1], ground_block[1][0],
                                 ground_block[1][1]))

    # Initialise pitfall instances
    pitfalls_list = []
    top_layer_blocks = [block for block in ground_blocks_list if block.top_y == 208]
    bottom_layer_blocks = [block for block in ground_blocks_list if block.top_y == 224]
    top_layer_gaps = []
    bottom_layer_gaps = []

    # Iterate through the top_layer_blocks and find gaps
    for i in range(len(top_layer_blocks) - 1):
        current_block = top_layer_blocks[i]
        next_block = top_layer_blocks[i + 1]

        # Check if there is a gap between the current and next block
        if next_block.left_x != current_block.right_x:
            top_layer_gaps.append((current_block.right_x, next_block.left_x))

    # Iterate through the bottom_layer_blocks and find gaps
    for i in range(len(bottom_layer_blocks) - 1):
        current_block = bottom_layer_blocks[i]
        next_block = bottom_layer_blocks[i + 1]

        # Check if there is a gap between the current and next block
        if next_block.left_x != current_block.right_x:
            bottom_layer_gaps.append((current_block.right_x, next_block.left_x))

    if top_layer_gaps and bottom_layer_gaps:
        for top_start, top_end in top_layer_gaps:
            for bottom_start, bottom_end in bottom_layer_gaps:
                if (top_start >= bottom_start and top_end <= bottom_end) or \
                        (top_start >= bottom_start and top_start <= bottom_end) or \
                        (top_end >= bottom_start and top_end <= bottom_end):
                    pitfalls_list.append(Pitfall(top_start, top_end - top_start))
                    break

    # Initialise question_block instances
    question_blocks_list = []
    if object_locations.get("block"):
        if object_locations.get("block").get("question_block"):
            for question_block in object_locations.get("block").get("question_block"):
                question_blocks_list.append(
                    Question_Block(question_block[0][0], question_block[0][1], question_block[1][0],
                                   question_block[1][1]))

    # Initialise pipe instances
    pipes_list = []
    if object_locations.get("block"):
        if object_locations.get("block").get("pipe"):
            for pipe in object_locations.get("block").get("pipe"):
                pipes_list.append(
                    Pipe(pipe[0][0], pipe[0][1], pipe[1][0], pipe[1][1]))

    # Initialise mushroom instances
    mushrooms_list = []
    if object_locations.get("item"):
        if object_locations.get("item").get("mushroom"):
            for mushroom in object_locations.get("item").get("mushroom"):
                mushrooms_list.append(
                    Mushroom(mushroom[0][0], mushroom[0][1], mushroom[1][0], mushroom[1][1]))
    
    # Intialise stair_block instances
    stair_blocks_list = []
    if object_locations.get("block"):
        if object_locations.get("block").get("stair_block"):
            for stair_block in object_locations.get("block").get("stair_block"):
                stair_blocks_list.append(
                    Stair_Block(stair_block[0][0], stair_block[0][1], stair_block[1][0], stair_block[1][1]))
                
     # Intialise brick_block instances
    brick_blocks_list = []
    if object_locations.get("block"):
        if object_locations.get("block").get("brick_block"):
            for brick_block in object_locations.get("block").get("brick_block"):
                brick_blocks_list.append(
                    Brick_Block(brick_block[0][0], brick_block[0][1], brick_block[1][0], brick_block[1][1]))
                
    ################ DEBUGGING PURPOSES ################
    if PRINT_GRID and step % 100 == 0:
        print_grid(screen, object_locations)
        print(object_locations)
    if PRINT_LOCATIONS and step == 700:
        print("------------- Step: ", step, " -------------")

        print("MARIO")
        if marios_list:
            mario = marios_list[0]
            print(f"\tMario: {(mario.left_x, mario.top_y)}), {(mario.width, mario.height)}")
        else:
            print("\tNone")

        print("\nENEMIES")
        if enemies_list:
            for enemy in enemies_list:
                print(f"\t{enemy.name}: {(enemy.left_x, enemy.top_y)}), {(enemy.width, enemy.height)}")
        else:
            print("\tNone")

        print("\nGROUND BLOCKS")
        if ground_blocks_list:
            for ground_block in ground_blocks_list:
                print(
                    f"\tGround Block: {(ground_block.left_x, ground_block.top_y)}), {(ground_block.width, ground_block.height)}")
        else:
            print("\tNone")

        print("\nQUESTION BLOCKS")
        if question_blocks_list:
            for question_block in question_blocks_list:
                print(
                    f"\tQuestion Block: {(question_block.left_x, question_block.top_y)}), {(question_block.width, question_block.height)}")
        else:
            print("\tNone")

        print("\nSTAIR BLOCKS")
        if stair_blocks_list:
            for stair_block in stair_blocks_list:
                print(
                    f"\tStair Block: {(stair_block.left_x, stair_block.top_y)}), {(stair_block.width, stair_block.height)}")
        else:
            print("\tNone")
        
        print("\nBRICK BLOCKS")
        if brick_blocks_list:
            for brick_block in brick_blocks_list:
                print(
                    f"\tBrick Block: {(brick_block.left_x, brick_block.top_y)}), {(brick_block.width, brick_block.height)}")
        else:
            print("\tNone")

        print("\nPIPES")
        if pipes_list:
            for pipe in pipes_list:
                print(f"\tPipe: {(pipe.left_x, pipe.top_y)}), {(pipe.width, pipe.height)}")
        else:
            print("\tNone")

        print("\nMUSHROOMS")
        if mushrooms_list:
            for mushroom in mushrooms_list:
                print(f"\tMushroom: {(mushroom.left_x, mushroom.top_y)}), {(mushroom.width, mushroom.height)}")
        else:
            print("\tNone")

        print("\nINFO DICTIONARY")
        print(info)

    ##################### RULE-BASED IMPLEMENTATION #####################
    if marios_list:
        mario = marios_list[0]

    # If there is at least one enemy in the screen, filter out enemies that are behind mario
    if enemies_list:
        enemies_in_front = [enemy for enemy in enemies_list if mario.right_x <= enemy.left_x]

        # If there is at least one enemy that is within 30 pixels from mario and at the same level as mario
        for enemy in enemies_in_front:
            if enemy.left_x - mario.right_x <= 30 and enemy.is_on_ground() and mario.is_on_ground():
                return RIGHT_A

    # If there is at least one pipe in the screen, filter out pipes that are behind mario
    if pipes_list:
        pipes_in_front = [pipe for pipe in pipes_list if mario.right_x <= pipe.left_x]

        # If there is at least one pipe that is within 30 pixels from mario and mario is on the ground
        for pipe in pipes_in_front:
            if pipe.left_x - mario.right_x <= 20 and pipe.is_on_ground() and mario.is_on_ground():

                # If pipe height is greater than mario's typical jump height
                if pipe.height > 60:
                    high_jump_counter = 2
                    return RIGHT_A
                else:
                    return RIGHT_A

    # If there is at least one pitfall in the screen, filter out pitfalls that are behind mario
    if pitfalls_list:
        pitfalls_in_front = [pitfall for pitfall in pitfalls_list if mario.right_x <= pitfall.left_x]

        # If there is at least one pipe that is within 30 pixels from mario and mario is on the ground
        for pitfall in pitfalls_in_front:
            if pitfall.left_x - mario.right_x <= 20 and mario.is_on_ground():
                return RIGHT_A
    
    # If there is at least one stair_block in the screen, filter out stair_blocks that are behind mario
    if stair_blocks_list:
        stairs_in_front = [stair_block for stair_block in stair_blocks_list if mario.right_x <= stair_block.left_x]

        # If there is at least one pipe that is within 30 pixels from mario and mario is on the ground
        for stair_block in stairs_in_front:
            if stair_block.left_x - mario.right_x <= 20 and -2 <= stair_block.bottom_y - mario.bottom_y <= 2:

                # If pipe height is greater than mario's typical jump height
                # if stair_block.height > 60:
                #     high_jump_counter = 2
                #     return RIGHT_A
                # else:
                return RIGHT_A

    return RIGHT


############################## MAIN #########################################

env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Change these values if you want more/less printing
PRINT_GRID = False
PRINT_LOCATIONS = True

# Actions List
NOOP = 0  # No operation
RIGHT = 1  # Move right only
RIGHT_A = 2  # Move right + Jump
RIGHT_B = 3  # Move right + Run
RIGHT_A_B = 4  # Move right + Jump + Run
A = 5  # Jump only
LEFT = 6  # Move left only

obs = None
done = True
prev_action = None
env.reset()

# Main Loop
for step in range(100000):
    if obs is None:
        action = env.action_space.sample()
    else:
        if step % 10 == 0:
            action = decide_action(obs, info)
        else:
            action = prev_action

    prev_action = action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if done:
        env.reset()

env.close()
