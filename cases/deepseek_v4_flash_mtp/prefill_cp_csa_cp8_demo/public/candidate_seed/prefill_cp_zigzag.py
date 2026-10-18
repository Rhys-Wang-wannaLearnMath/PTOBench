from config import TAIL_ROWS, CP_SIZE, NUM_SEGMENTS

from config import CASE_CONFIG as M


HEAD_DIM = M.head_dim

EPOCHS = 1

ROW_TILE = 8



CP_TAIL_WINDOW_ROWS = NUM_SEGMENTS * TAIL_ROWS
