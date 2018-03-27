import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from matplotlib.widgets import RectangleSelector
from generate_json import write_json
    
class Box:
    upper_left = None
    down_right = None
    piece_colour = None
    field_colour = None
    piece = None
    
    def __str__(self):
        return str(self.piece) + " " + str(self.colour) + " " + str(self.upper_left) + " " + str(self.down_right)
 
# global constants
img = None
boxes = []

# constants
image_folder = 'my_photos'

# variables
piece = "EMPTY"
field_colour = "WHITE"
piece_colour = "WHITE"

field_white_patch = mpatches.Patch(color='GRAY', label='field colour: WHITE')
field_black_patch = mpatches.Patch(color='BLACK', label='field colour: BLACK')
piece_white_patch = mpatches.Patch(color='GRAY', label='piece colour: WHITE')
piece_black_patch = mpatches.Patch(color='BLACK', label='piece colour: BLACK')
piece_empty_patch = mpatches.Patch(color='RED', label='no piece here')
empty_patch = mpatches.Patch(color='GRAY', label='piece: EMPTY')
queen_patch = mpatches.Patch(color='RED', label='piece: QUEEN')
king_patch = mpatches.Patch(color='YELLOW', label='piece: KING')
rook_patch = mpatches.Patch(color='BLUE', label='piece: ROOK')
bishop_patch = mpatches.Patch(color='GREEN', label='piece: BISHOP')
knight_patch = mpatches.Patch(color='BROWN', label='piece: KNIGHT')
pawn_patch = mpatches.Patch(color='PINK', label='piece: PAWN')

piece_patch = empty_patch
piece_colour_patch = piece_empty_patch
field_colour_patch = field_white_patch

def line_select_callback(clk, rls):
    global boxes
    global piece
    global field_colour
    global piece_colour
    box = Box()
    box.upper_left = (int(clk.xdata), int(clk.ydata))
    box.down_right = (int(rls.xdata), int(rls.ydata))
    box.field_colour = field_colour
    box.piece_colour = piece_colour
    box.piece = piece
    boxes.append(box)


def on_key(event):
    global img
    global boxes
    global piece
    global field_colour
    global piece_colour
    global piece_colour_patch
    global field_colour_patch
    global piece_patch
    if event.key == 'c':
        field_colour = "WHITE"
        field_colour_patch = field_white_patch
        update_legend()
    if event.key == 'd':
        field_colour = "BLACK"
        field_colour_patch = field_black_patch
        update_legend()
    if event.key == 'z':
        piece_colour = "WHITE"
        piece_colour_patch = piece_white_patch
        update_legend()
    if event.key == 'a':
        piece_colour = "BLACK"
        piece_colour_patch = piece_black_patch
        update_legend()
    if event.key == '1':
        piece = "ROOK"
        piece_patch = rook_patch
        update_legend()
    if event.key == '2':
        piece = "KNIGHT"
        piece_patch = knight_patch
        update_legend()
    if event.key == '3':
        piece = "BISHOP"
        piece_patch = bishop_patch
        update_legend()
    if event.key == '4':
        piece = "QUEEN"
        piece_patch = queen_patch
        update_legend()
    if event.key == '5':
        piece = "KING"
        piece_patch = king_patch
        update_legend()
    if event.key == '6':
        piece = "PAWN"
        piece_patch = pawn_patch
        update_legend()
    if event.key == '7':
        piece = "EMPTY"
        piece_colour = "EMPTY"
        piece_patch = empty_patch
        piece_colour_patch = piece_empty_patch
        update_legend()        
    if event.key == 'q':
        write_json(img, boxes)
        piece_patch = empty_patch
        field_colour_patch = field_white_patch
        piece_colour_patch = piece_white_patch
        boxes = []
        img = None
        plt.close()
        
def update_legend():
    plt.legend(handles=[field_colour_patch, piece_colour_patch, piece_patch], bbox_to_anchor=(0., 1.02, 1., .102), loc=4, ncol=1, mode="expand", borderaxespad=0.)
    plt.draw()
    
def toggle_selector(event):
    toggle_selector.RS.set_active(True)


if __name__ == '__main__':
    for n, image_file in enumerate(os.scandir(image_folder)):
        img = image_file
        fig, ax = plt.subplots(1)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(250, 120, 1280, 1024)
        
        image = cv2.imread(image_file.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)

        toggle_selector.RS = RectangleSelector(
            ax, line_select_callback,
            drawtype='box', useblit=True,
            button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=False
        )

        key = plt.connect('key_press_event', on_key)
        bbox = plt.connect('key_press_event', toggle_selector)
        update_legend()
        
        plt.show()