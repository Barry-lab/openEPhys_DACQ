# This is a task

import pygame
import numpy as np
import threading
import RewardController as Rewards
import time
from scipy.spatial.distance import euclidean
import random

def smooth_edge_padding(data, smoothing):
    originalSize = data.size
    data = np.convolve(data, np.ones((smoothing,))/smoothing, mode='valid')
    missing = originalSize - data.size
    addStart = int(np.floor(missing / 2.0))
    addEnd = int(np.ceil(missing / 2.0))
    data = np.lib.pad(data, (addStart, addEnd), 'edge')

    return data



def compute_distance_travelled(posHistory, smoothing):
    distances = []
    posHistory = np.array(posHistory)[:, :2]
    posHistory[:,0] = smooth_edge_padding(posHistory[:,0], smoothing)
    posHistory[:,1] = smooth_edge_padding(posHistory[:,1], smoothing)
    for npos in range(posHistory.shape[0] - 1):
        prev_pos = posHistory[npos, :]
        curr_pos = posHistory[npos + 1, :]
        distance = euclidean(prev_pos, curr_pos)
        distances.append(distance)
    total_distance = np.sum(np.array(distances))

    return total_distance

class Core(object):
    def __init__(self, TaskSettings, TaskIO):
        self.TaskIO = TaskIO
        self.TaskSettings = TaskSettings
        self.TaskSettings['LastTravelTime'] = 2
        self.TaskSettings['LastTravelDist'] = 20
        self.TaskSettings['LastTravelSmooth'] = 3
        self.TaskSettings['XRange'] = [80, 200]
        self.TaskSettings['YRange'] = [60, 130]
        self.TaskSettings['LastTravelDist'] = 50
        self.TaskSettings['Chewing_TTLchan'] = 5
        self.TaskSettings['Chewing_Target'] = 10
        self.TaskSettings['PelletRewardMinSeparation'] = 10
        self.TaskSettings['PelletRewardMaxSeparation'] = 90
        self.TaskSettings['UsePelletFeeders'] = [2]
        self.TaskSettings['UseMilkFeeders'] = [1]
        self.TaskSettings['InitPelletsReward'] = 5
        self.TaskSettings['InitMilkReward'] = 2.0
        self.TaskSettings['PelletRewardSize'] = 1
        # Pre-compute variables
        self.one_second_steps = int(np.round(1 / self.TaskIO['RPIPos'].update_interval))
        self.distance_steps = int(np.round(self.TaskSettings['LastTravelTime'])) * self.one_second_steps
        # Prepare TTL pulse time list
        self.ttlTimes = []
        self.ttlTimesLock = threading.Lock()
        self.TaskIO['OEmessages'].add_callback(self.append_ttl_pulses)
        # Prepare Pellet rewards
        self.PelletReward = []
        for n_Pfeeder in self.TaskSettings['UsePelletFeeders']:
            self.PelletReward.append(Rewards.Pellets(n_Pfeeder))
        self.lastPelletReward = 0
        self.pelletRewardsOn = True
        # Prepare Milk rewards
        self.MilkReward = []
        for n_Mfeeder in self.TaskSettings['UseMilkFeeders']:
            self.MilkReward.append(Rewards.Milk(n_Mfeeder))
        self.lastMilkReward = 0
        self.milkRewardsOn = True
        # Ensure that Position data is available
        posHistory = []
        while len(posHistory) < self.distance_steps:
            time.sleep(0.5)
            with self.TaskIO['RPIPos'].combPosHistoryLock:
                posHistory = self.TaskIO['RPIPos'].combPosHistory
        # Set game speed
        self.responseRate = 60 # Hz
        self.gameRate = 10 # Hz
        # Initialize game
        self.gameOn = True
        self.mainLoopActive = True
        self.clock = pygame.time.Clock()
        pygame.init()

    def renderText(self, text):
        renderedText = self.font.render(text, True, self.textColor)

        return renderedText

    def append_ttl_pulses(self, message):
        parts = message.split()
        if parts[2] == str(self.TaskSettings['Chewing_TTLchan']):
            with self.ttlTimesLock:
                self.ttlTimes.append(time.time())

    def number_of_chewings(self, lastReward):
        with self.ttlTimesLock:
            chewing_times = self.ttlTimes
        n_chewings = np.sum(np.array(chewing_times) > lastReward)

        return n_chewings

    def releasePellet(self, n_Pfeeder, actor='GameEvent', n_pellets=1):
        # Release Pellet
        feeder_list_idx = self.TaskSettings['UsePelletFeeders'].index(n_Pfeeder)
        self.PelletReward[feeder_list_idx].release(n_pellets)
        self.lastPelletReward = time.time()
        # Send message to Open Ephys GUI
        OEmessage = 'PelletReward ' + actor + ' ' + str(n_Pfeeder + 1) + ' ' + str(n_pellets)
        self.TaskIO['MessageToOE'](OEmessage)

    def depositMilk(self, n_Mfeeder, actor='GameEvent', quantity=1.0):
        # Release Pellet
        feeder_list_idx = self.TaskSettings['UseMilkFeeders'].index(n_Mfeeder)
        self.MilkReward[feeder_list_idx].release(quantity)
        self.lastMilkReward = time.time()
        # Send message to Open Ephys GUI
        OEmessage = 'MilkReward ' + actor + ' ' + str(n_Mfeeder + 1) + ' ' + str(quantity)
        self.TaskIO['MessageToOE'](OEmessage)

    def initRewards(self):
        if 'InitPelletsReward' in self.TaskSettings.keys() and self.TaskSettings['InitPelletsReward'] > 0:
            minPellets = int(np.floor(float(self.TaskSettings['InitPelletsReward']) / len(self.TaskSettings['UsePelletFeeders'])))
            extraPellets = np.mod(self.TaskSettings['InitPelletsReward'], len(self.TaskSettings['UsePelletFeeders']))
            n_pellets_Feeders = minPellets * np.ones(len(self.TaskSettings['UsePelletFeeders']), dtype=np.int16)
            n_pellets_Feeders[:extraPellets] = n_pellets_Feeders[:extraPellets] + 1
            for feeder_list_idx, n_pellets in enumerate(n_pellets_Feeders):
                n_Pfeeder = self.TaskSettings['UsePelletFeeders'][feeder_list_idx]
                self.releasePellet(n_Pfeeder, actor='GameInit', n_pellets=n_pellets)

    def buttonGameOnOff_callback(self):
        # Switch Game On and Off
        self.gameOn = not self.gameOn
        OEmessage = 'Game On: ' + str(self.gameOn)
        self.TaskIO['MessageToOE'](OEmessage)

    def buttonReleasePellet_callback(self, n_Pfeeder):
        # Release pellet from specified feeder and mark as User action
        self.releasePellet(n_Pfeeder, actor='User')

    def buttonManualPellet_callback(self):
        # Update last reward time
        self.lastPelletReward = time.time()
        # Send message to Open Ephys GUI
        OEmessage = 'PelletReward Manual'
        self.TaskIO['MessageToOE'](OEmessage)

    def defineButtons(self):
        # Add or remove buttons from this function
        # Note default settings applied in self.createButtons()
        buttons = []
        # Game On/Off button
        buttonGameOnOff = {'callback': self.buttonGameOnOff_callback, 
                           'text': 'Game On', 
                           'toggled': {'text': 'Game Off', 
                                       'color': (0, 0, 255)}}
        buttons.append(buttonGameOnOff)
        # Button to release pellet
        buttonReleasePellet = []
        buttonReleasePellet.append({'text': 'Release Pellet'})
        for n_Pfeeder in self.TaskSettings['UsePelletFeeders']:
            nFeederButton = {'callback': self.buttonReleasePellet_callback, 
                             'callargs': [n_Pfeeder], 
                             'text': str(n_Pfeeder + 1)}
            buttonReleasePellet.append(nFeederButton)
        buttons.append(buttonReleasePellet)
        # Button to mark manually released pellet
        buttonManualPellet = {'callback': self.buttonManualPellet_callback, 
                              'text': 'Manual Pellet'}
        buttons.append(buttonManualPellet)

        return buttons

    def createButtons(self):
        buttons = self.defineButtons()
        # Add default color to all buttons
        for i, button in enumerate(buttons):
            if isinstance(button, dict):
                if not 'color' in button.keys():
                    buttons[i]['color'] = (128, 128, 128)
            elif isinstance(button, list):
                for j, subbutton in enumerate(button[1:]):
                    if not 'color' in subbutton.keys():
                        buttons[i][j + 1]['color'] = (128, 128, 128)
        # Add default button un-pressed state
        for i, button in enumerate(buttons):
            if isinstance(button, dict):
                if not 'button_pressed' in button.keys():
                    buttons[i]['button_pressed'] = False
            elif isinstance(button, list):
                for j, subbutton in enumerate(button[1:]):
                    if not 'button_pressed' in subbutton.keys():
                        buttons[i][j + 1]['button_pressed'] = False
        # Compute button locations
        xpos = self.screen_size[0] - self.screen_size[0] * self.buttonProportions + self.screen_margins
        xlen = self.screen_size[0] * self.buttonProportions - 2 * self.screen_margins
        ypos = np.linspace(self.screen_margins, self.screen_size[1] - self.screen_margins, 2 * len(buttons))
        ylen = ypos[1] - ypos[0]
        ypos = ypos[::2]
        for i, button in enumerate(buttons):
            if isinstance(button, dict):
                buttons[i]['position'] = (int(round(xpos)), int(round(ypos[i])), int(round(xlen)), int(round(ylen)))
            elif isinstance(button, list):
                xsubpos = np.linspace(xpos, xpos + xlen, 2 * (len(button) - 1))
                xsublen = xsubpos[1] - xsubpos[0]
                xsubpos = xsubpos[::2]
                for j, subbutton in enumerate(button):
                    if j == 0:
                        buttons[i][j]['position'] = (int(round(xpos)), int(round(ypos[i])), int(round(xlen)), int(round(ylen / 2.0)))
                    else:
                        buttons[i][j]['position'] = (int(round(xsubpos[j - 1])), int(round(ypos[i] + ylen / 2.0)), int(round(xsublen)), int(round(ylen / 2.0)))
        # Create button rectangles
        for i, button in enumerate(buttons):
            if isinstance(button, dict):
                buttons[i]['Rect'] = pygame.Rect(button['position'])
            elif isinstance(button, list):
                for j, subbutton in enumerate(button[1:]):
                    buttons[i][j + 1]['Rect'] = pygame.Rect(subbutton['position'])
        # Render button texts
        for i, button in enumerate(buttons):
            if isinstance(button, dict):
                buttons[i]['textRendered'] = self.renderText(button['text'])
                if 'toggled' in button.keys():
                    buttons[i]['toggled']['textRendered'] = self.renderText(button['toggled']['text'])
            elif isinstance(button, list):
                for j, subbutton in enumerate(button):
                    buttons[i][j]['textRendered'] = self.renderText(subbutton['text'])

        return buttons

    def draw_buttons(self):
        # Draw all buttons here
        # When drawing buttons in Pushed state color, reset the state unless Toggled in keys
        for i, button in enumerate(self.buttons):
            if isinstance(button, dict):
                textRendered = button['textRendered']
                if button['button_pressed']:
                    if 'toggled' in button.keys():
                        color = button['toggled']['color']
                        textRendered = button['toggled']['textRendered']
                    else:
                        self.buttons[i]['button_pressed'] = False
                        color = (int(button['color'][0] * 0.5), int(button['color'][1] * 0.5), int(button['color'][2] * 0.5))
                else:
                    color = button['color']
                pygame.draw.rect(self.screen, color, button['position'], 0)
                self.screen.blit(textRendered, button['position'][:2])
            elif isinstance(button, list):
                for j, subbutton in enumerate(button[1:]):
                    if subbutton['button_pressed']:
                        self.buttons[i][j + 1]['button_pressed'] = False
                        color = (int(subbutton['color'][0] * 0.5), int(subbutton['color'][1] * 0.5), int(subbutton['color'][2] * 0.5))
                    else:
                        color = subbutton['color']
                    pygame.draw.rect(self.screen, subbutton['color'], subbutton['position'], 0)
                    self.screen.blit(subbutton['textRendered'], subbutton['position'][:2])

    def create_progress_bars(self):
        game_progress = self.game_logic()
        # Initialise random color generation
        random.seed(123)
        rcolor = lambda: random.randint(0,255)
        # Compute progress bar locations
        textSpacing = 2
        textSpace = 10 + textSpacing
        textLines = 3
        xpos = np.linspace(self.screen_margins, self.screen_size[0] * (1 - self.buttonProportions) - self.screen_margins, 2 * len(game_progress))
        xlen = xpos[1] - xpos[0]
        xpos = xpos[::2]
        ybottompos = self.screen_size[1] - self.screen_margins - textSpace * textLines - textSpacing
        ymaxlen = self.screen_size[1] - 2 * self.screen_margins - textSpace * textLines
        progress_bars = []
        for i, gp in enumerate(game_progress):
            progress_bars.append({'name_text': self.renderText(gp['name']), 
                                       'target_text': self.renderText('Target: ' + str(gp['target'])), 
                                       'value_text': '', 
                                       'name_position': (xpos[i], ybottompos + 0 * textSpace), 
                                       'target_position': (xpos[i], ybottompos + 1 * textSpace), 
                                       'value_position': (xpos[i], ybottompos + 2 * textSpace), 
                                       'color': (rcolor(), rcolor(), rcolor()), 
                                       'position': {'xpos': xpos[i], 
                                                    'xlen': xlen, 
                                                    'ybottompos': ybottompos, 
                                                    'ymaxlen': ymaxlen}})

        return progress_bars

    def draw_progress_bars(self, game_progress):
        for gp, pb in zip(game_progress, self.progress_bars):
            self.screen.blit(pb['name_text'], pb['name_position'])
            self.screen.blit(pb['target_text'], pb['target_position'])
            self.screen.blit(self.renderText(str(gp['status'])), pb['value_position'])
            if gp['complete']:
                color = (255, 255, 255)
            else:
                color = pb['color']
            ylen = int(round(gp['percentage'] * pb['position']['ymaxlen']))
            ypos = pb['position']['ybottompos'] - ylen
            position = (pb['position']['xpos'], ypos, pb['position']['xlen'], ylen)
            pygame.draw.rect(self.screen, color, position, 0)

    def game_logic(self):
        game_progress = []
        # Get animal position history
        with self.TaskIO['RPIPos'].combPosHistoryLock:
            posHistory = self.TaskIO['RPIPos'].combPosHistory[-self.distance_steps:]
        # Check if enough time as passed since last pellet reward
        timeSinceLastReward = time.time() - self.lastPelletReward
        rewardDelayCriteria = {'name': 'Since Reward', 
                               'goals': ['pellet'], 
                               'target': self.TaskSettings['PelletRewardMinSeparation'], 
                               'status': int(round(timeSinceLastReward)), 
                               'complete': timeSinceLastReward >= self.TaskSettings['PelletRewardMinSeparation'], 
                               'percentage': timeSinceLastReward / float(self.TaskSettings['PelletRewardMinSeparation'])}
        game_progress.append(rewardDelayCriteria)
        # Check if has been moving enough in the last few seconds
        total_distance = compute_distance_travelled(posHistory, self.TaskSettings['LastTravelSmooth'])
        mobilityCriteria = {'name': 'Mobility', 
                            'goals': ['pellet'], 
                            'target': self.TaskSettings['LastTravelDist'], 
                            'status': int(round(total_distance)), 
                            'complete': total_distance >= self.TaskSettings['LastTravelDist'], 
                            'percentage': total_distance / float(self.TaskSettings['LastTravelDist'])}
        game_progress.append(mobilityCriteria)
        # Check if animal has been chewing enough since last reward
        n_chewings = self.number_of_chewings(self.lastPelletReward)
        chewingCriteria = {'name': 'Chewing', 
                            'goals': ['pellet'], 
                           'target': self.TaskSettings['Chewing_Target'], 
                           'status': n_chewings, 
                           'complete': n_chewings >= self.TaskSettings['Chewing_Target'], 
                           'percentage': n_chewings / float(self.TaskSettings['Chewing_Target'])}
        game_progress.append(chewingCriteria)
        # Check if animal has been without pellet reward for too long
        timeSinceLastReward = time.time() - self.lastPelletReward
        rewardDelayCriteria = {'name': 'Inactivity', 
                               'goals': ['inactivity'], 
                               'target': self.TaskSettings['PelletRewardMaxSeparation'], 
                               'status': int(round(timeSinceLastReward)), 
                               'complete': timeSinceLastReward >= self.TaskSettings['PelletRewardMaxSeparation'], 
                               'percentage': timeSinceLastReward / float(self.TaskSettings['PelletRewardMaxSeparation'])}
        game_progress.append(rewardDelayCriteria)
        # If all ok, release reward
        pellet_status_complete = []
        inactivity_complete = []
        for gp in game_progress:
            if 'pellet' in gp['goals']:
                pellet_status_complete.append(gp['complete'])
            if 'inactivity' in gp['goals']:
                inactivity_complete.append(gp['complete'])
        if all(pellet_status_complete):
            for n_Pfeeder in self.TaskSettings['UsePelletFeeders']:
                self.releasePellet(n_Pfeeder, actor='Goal_pellet', n_pellets=1)
        elif all(inactivity_complete):
            n_Pfeeder = self.TaskSettings['UsePelletFeeders'][random.randint(0,len(self.TaskSettings['UsePelletFeeders']) - 1)]
            self.releasePellet(n_Pfeeder, actor='Goal_inactivity', n_pellets=1)

        return game_progress

    def update_display(self):
        self.screen.fill((0, 0, 0))
        self.draw_buttons()
        self.draw_progress_bars(self.game_progress)
        # Update display
        pygame.display.update()


    def update_states(self):
        if self.gameOn:
            self.game_progress = self.game_logic()
        self.update_display()

    def main_loop(self):
        # Initialize interactive elements
        self.screen_size = (600, 300)
        self.screen_margins = 10
        self.buttonProportions = 0.2
        self.font = pygame.font.SysFont('Arial', 10)
        self.textColor = (255, 255, 255)
        self.screen = pygame.display.set_mode(self.screen_size)
        self.buttons = self.createButtons()
        self.progress_bars = self.create_progress_bars()
        # Signal game start to Open Ephys GUI
        OEmessage = 'Game On: ' + str(self.gameOn)
        self.TaskIO['MessageToOE'](OEmessage)
        self.initRewards()
        lastUpdatedState = 0
        while self.mainLoopActive:
            self.clock.tick(self.responseRate)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.mainLoopActive = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        for i, button in enumerate(self.buttons):
                            if isinstance(button, dict) and button['Rect'].collidepoint(event.pos):
                                if 'toggled' in button.keys():
                                    self.buttons[i]['button_pressed'] = not button['button_pressed']
                                else:
                                    self.buttons[i]['button_pressed'] = True
                                if 'callargs' in button.keys():
                                    button['callback'](*button['callargs'])
                                else:
                                    button['callback']()
                            elif isinstance(button, list):
                                for j, subbutton in enumerate(button[1:]):
                                    if subbutton['Rect'].collidepoint(event.pos):
                                        self.buttons[i][j + 1]['button_pressed'] = True
                                        if 'callargs' in subbutton.keys():
                                            subbutton['callback'](*subbutton['callargs'])
                                        else:
                                            subbutton['callback']()
            if lastUpdatedState < (time.time() - 1 / float(self.gameRate)):
                lastUpdatedState = time.time()
                # TO DO! Make sure somehow that this thread has finished before updating
                # Show an error of thread is running too slowly
                threading.Thread(target=self.update_states).start()
                
        # Quite game when out of gameOn loop
        pygame.quit()

    def run(self):
        threading.Thread(target=self.main_loop).start()

    def stop(self):
        self.mainLoopActive = False
