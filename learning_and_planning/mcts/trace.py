from collections import namedtuple
import pickle

import attr
from gym_sokoban.envs import SokobanEnv
from gym_sokoban.envs.render_utils import render_state
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ACTION_TO_ASCII = ('^', 'v', '<', '>', '.^', '.v', '.<', '.>')


def expand(array, size, axis):
    shape = list(array.shape)
    shape[axis] = size - array.shape[axis]
    return np.concatenate([array, np.full(shape, 0)])


@attr.s
class Trace(object):
    seed = attr.ib(default=0)
    steps = attr.ib(factory=list)

    def render(self):
        rendered_steps = [step.render() for step in self.steps]
        (_, _, num_channels) = rendered_steps[0].shape
        max_step_height = max(
            rendered_step.shape[0] for rendered_step in rendered_steps
        )
        delimiter = np.full((max_step_height, 2, num_channels), 255)
        delimiter[:, 0, :] = 0
        rendered_trace = np.concatenate([
            x
            for rendered_steps in rendered_steps
            for x in (expand(rendered_steps, max_step_height, axis=0), delimiter)
        ], axis=1)

        (_, trace_width, _) = rendered_trace.shape
        seed_bar_height = 10
        seed_bar = Image.new('RGB', (trace_width, seed_bar_height))
        draw = ImageDraw.Draw(seed_bar)
        seed_str = 'seed: {}'.format(hex(self.seed))
        draw.text((0, 0), seed_str, font=get_font(size=8))
        rendered_trace[:seed_bar_height, :, :] = np.asarray(seed_bar)
        return rendered_trace


@attr.s
class Step(object):
    action = attr.ib(default=None)
    passes = attr.ib(factory=list)

    def render(self):
        rendered_passes = [pass_.render() for pass_ in self.passes]
        (_, _, num_channels) = rendered_passes[0].shape
        max_pass_height = max(
            rendered_pass.shape[0] for rendered_pass in rendered_passes
        )
        delimiter = np.full((max_pass_height, 1, num_channels), 255)
        rendered_passes = np.concatenate([
            x
            for rendered_pass in rendered_passes
            for x in (expand(rendered_pass, max_pass_height, axis=0), delimiter)
        ], axis=1)

        (_, passes_width, _) = rendered_passes.shape
        (node_height, _, _) = self.passes[0].nodes[0].render().shape
        action_bar = Image.new('RGB', (passes_width, node_height))
        draw = ImageDraw.Draw(action_bar)
        draw.text(
            ((passes_width - node_height) // 2, 0),
            ACTION_TO_ASCII[self.action],
            font=get_font(size=node_height),
        )
        return np.concatenate([np.asarray(action_bar), rendered_passes], axis=0)


@attr.s
class Pass(object):
    nodes = attr.ib(factory=list)

    def render(self):
        rendered_nodes = [node.render() for node in self.nodes]
        (_, node_width, num_channels) = rendered_nodes[0].shape
        delimiter = np.full((1, node_width, num_channels), 255)
        return np.concatenate([
            x
            for rendered_node in rendered_nodes
            for x in (rendered_node, delimiter)
        ], axis=0)


@attr.s
class Node(object):
    state = attr.ib()
    num_visits = attr.ib()
    value = attr.ib()
    children = attr.ib(factory=dict)
    action = attr.ib(default=None)

    def render(self):
        board = render_state(self.state)
        (board_height, board_width, _) = board.shape
        font = get_font(size=8)
        info_bar = Image.new('RGB', (27, board_height))
        draw = ImageDraw.Draw(info_bar)

        def format_float(x):
            """Space-efficient float formatting.

            Limits precision to two digits after the comma. Omits the leading
            zero for numbers less than 1 (e.g. .3). Omits the fractional part
            for numbers greater than 1.
            """
            s = '{:.2f}'.format(x).rstrip('0')
            neg = False
            if s[0] == '-':
                s = s[1:]
                neg = True
            if s[0] == '0' and s[-1] != '.':
                s = s[1:]
            else:
                s = s[:s.index('.')]
            if neg:
                s = '-' + s
            return s

        default_color = (255, 255, 255, 255)  # white
        active_color = (0, 255, 0, 255)  # green
        seen_color = (255, 0, 0, 255)  # red
        def child_color(action):
            if action == self.action:
                return active_color
            elif self.children[action].seen:
                return seen_color
            else:
                return default_color

        for (row, (label, value, color)) in enumerate(
            [
                ('v', self.value, default_color),
                ('c', self.num_visits, default_color),
            ] + [
                (label, self.children[action].value, child_color(action))
                for (action, label) in enumerate(ACTION_TO_ASCII)
                if action in self.children
            ]
        ):
            info = '{}:{}'.format(label, format_float(value))
            draw.text((0, 8 * row), info, font=font, fill=color)

        return np.concatenate([np.asarray(info_bar), board], axis=1)


Child = namedtuple('Child', ('value', 'seen'))


FONTS = {}


def get_font(size):
    if size not in FONTS:
        FONTS[size] = ImageFont.truetype(
            'assets/slkscr.ttf', size=size
        )
    return FONTS[size]
