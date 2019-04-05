import sc2, random
from sc2 import run_game, maps, Race, Difficulty, Result, position
from sc2.player import Bot, Computer, Human
from sc2.constants import *
import cv2, time
import numpy as np
import keras

HEADLESS = False

class SentdeBot(sc2.BotAI):
    def __init__(self, use_model=False):
        self.MAX_SCV = 60
        self.LIMIT_supply = 200
        self.iteration_scout = 0
        self.scv_scouting = 0
        self.iteration_per_min = 165
        self.do_something_after = 0
        self.train_data = []
        self.use_model = use_model
        self.nb_cc = 3
        self.base_location = []
        self.barrack_try_addon = []


        if self.use_model:
            print("USING MODEL!")
            self.model = keras.models.load_model("Model-100-epochs-0.0001-attack")

        #Exemple -> COMMANDCENTER : 3 = sight, (0,255,0) = RGB
        self.sight_def = {
                        COMMANDCENTER: [11],
                        ORBITALCOMMAND: [11],
                        SUPPLYDEPOT: [9],
                        SUPPLYDEPOTLOWERED: [9],
                        SCV: [8],
                        MARINE: [9],
                        REFINERY: [9],
                        BARRACKS: [9],
                    }

        #Exemple -> COMMANDCENTER : 3 = size, (0,255,0) = RGB
        self.size_def = {
                        COMMANDCENTER: [3, (0, 255, 0)],
                        ORBITALCOMMAND: [3, (0, 255, 0)],
                        SUPPLYDEPOT: [1, (20, 235, 0)],
                        SUPPLYDEPOTLOWERED: [1, (20, 235, 0)],
                        SCV: [1, (55, 200, 0)],
                        MARINE: [1, (55, 200, 0)],
                        REFINERY: [1, (55, 200, 0)],
                        BARRACKS: [2, (200, 100, 0)],
                    }

        self.army_units = {
                    MARINE,
                    MARAUDER,
                    HELLION,
                }

    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result, self.use_model)

        with open("log.txt","a") as f:
            if self.use_model:
                f.write("Model {}\n".format(game_result))
            else:
                f.write("Random {}\n".format(game_result))

        if game_result == Result.Victory:
            np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))

    async def on_step(self, iteration):
        self.iteration=iteration

        for el in self.expansion_locations:
            self.base_location.append(el)

        await self.outputRGB()
        await self.distribute_workers()
        await self.build_workers() #Build scv
        await self.build_supply() #Build supply to increase the limit of population
        await self.morph_cc_in_orbital()
        await self.mule_or_scan()
        await self.down_supply()
        await self.build_gaz()
        await self.scout()
        await self.expand()
        await self.build_offensive_structure()
        await self.build_offensive_unit()
        await self.defend()
        await self.attack()
        await self.reset_scout()
        await self.build_research_building()
        await self.upgrade()

    def is_in_list(self, tag):
        ok = False
        i = 0
        if len(self.barrack_try_addon):
            while i < len(self.barrack_try_addon) and not ok:
                ok = (self.barrack_try_addon[i]==tag)
                i += 1

        return ok

    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += ((random.randrange(-20, 20))/100) * enemy_start_location[0]
        y += ((random.randrange(-20, 20))/100) * enemy_start_location[1]

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))

        return go_to

    async def scout(self):
        if len(self.units(SCV)) > 0:
            scout = self.units(SCV)[0]
            if scout.is_idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_location_variance(enemy_location)
                await self.do(scout.move(move_to))

    async def outputRGB(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        for unit_sight in self.sight_def:
            for i in self.units(unit_sight).ready:
                pos = i.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), self.sight_def[unit_sight][0],(255,250,250), -1)

        for unit_size in self.size_def:
            for i in self.units(unit_size).ready:
                pos = i.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), self.size_def[unit_size][0], self.size_def[unit_size][1], -1)

        for cc in self.units.of_type([COMMANDCENTER, ORBITALCOMMAND]).ready:
            nearby_minerals = self.state.mineral_field.closer_than(10, cc)
            for mineral in nearby_minerals:
                pos = mineral.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255,0,0), -1)

        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 2, (0, 0, 255), -1)

        for enemy_unit in self.known_enemy_units.not_structure:
            pos = enemy_unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (0, 0, 255), -1)

        line_max = 50
        mineral_count = self.minerals

        vespene_count = self.vespene
        plausible_supply = 0
        population_ratio = 0
        if self.supply_left > 0 and self.supply_cap > 0:
            population_ratio = self.supply_left / self.supply_cap
        else:
            population_ratio = 0

        if self.supply_cap > 0:
            plausible_supply = self.supply_cap / 200.0
        else:
            plausible_supply = 0

        pop = self.supply_cap-self.supply_left
        if pop == 0:
            pop = 200
        military_weight = self.units.of_type(self.army_units).amount / pop
        if military_weight > 1.0:
            military_weight = 1.0


        cv2.line(game_data, (0, 19), (int(line_max*military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
        cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 7), (int(vespene_count/line_max), 7), (210, 200, 0), 3)  # gas / 1500
        cv2.line(game_data, (0, 3), (int(mineral_count/line_max), 3), (0, 255, 25), 3)  # minerals minerals/1500

        self.flipped = cv2.flip(game_data, 0)

        if not HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow('Intel', resized)
            cv2.waitKey(1)

    async def build_workers(self):
        if self.units(SCV).amount < self.MAX_SCV:
            for cc in self.units.of_type([COMMANDCENTER, ORBITALCOMMAND]).ready.noqueue:
                if self.can_afford(SCV) and not self.can_afford(ORBITALCOMMAND):
                    await self.do(cc.train(SCV))

    async def mule_or_scan(self):
        if self.iteration < 900:
            await self.drop_mule()
        else:
            n = random.randrange(0,5)
            if n < 2:
                await self.drop_mule()
            else:
                await self.scan()

    async def drop_mule(self):
        for oc in self.units(ORBITALCOMMAND).ready:
            abilities = await self.get_available_abilities(oc)
            if CALLDOWNMULE_CALLDOWNMULE in abilities:
                for cc_alt in self.units(ORBITALCOMMAND).ready:
                    pack_minerals = self.state.mineral_field.closer_than(10.0,cc_alt)
                    for minerals in pack_minerals:
                        await self.do(oc(CALLDOWNMULE_CALLDOWNMULE,minerals))

    async def scan(self):
        for oc in self.units(ORBITALCOMMAND).ready:
            abilities = await self.get_available_abilities(oc)
            if CALLDOWNMULE_CALLDOWNMULE in abilities:
                for cc_alt in self.units(ORBITALCOMMAND).ready:
                    n = random.randrange(0,len(self.base_location))
                    await self.do(oc(SCANNERSWEEP_SCAN,self.base_location[n]))

    async def build_supply(self):
        if self.supply_left < 3 and not self.already_pending(SUPPLYDEPOT) and self.supply_used < self.LIMIT_supply:
            cc_onMap = self.units.of_type([COMMANDCENTER, ORBITALCOMMAND]).ready
            if (cc_onMap.exists):
                if self.can_afford(SUPPLYDEPOT):
                    await self.build(SUPPLYDEPOT, near = cc_onMap.first)
        if self.supply_left <= 0 and self.already_pending(SUPPLYDEPOT) <= 2 and self.supply_used < self.LIMIT_supply:
            cc_onMap = self.units.of_type([COMMANDCENTER, ORBITALCOMMAND]).ready
            if (cc_onMap.exists):
                if self.can_afford(SUPPLYDEPOT):
                    await self.build(SUPPLYDEPOT, near = cc_onMap.first)

    async def down_supply(self):
        for supp_up in self.units(SUPPLYDEPOT).ready:
            if self.can_afford(MORPH_SUPPLYDEPOT_LOWER):
                await self.do(supp_up(MORPH_SUPPLYDEPOT_LOWER))

    async def morph_cc_in_orbital(self):
        for cc in self.units(COMMANDCENTER).ready.idle:
            if self.can_afford(ORBITALCOMMAND) and len(self.units(BARRACKS).ready) > 1:
                await self.do(cc(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND))

    async def build_gaz(self):
        if self.supply_used >= 15:
            if self.units(REFINERY).amount <1 or (self.units(REFINERY).amount >= 1 and self.units(BARRACKS).amount >= 1):
                for cc in self.units(COMMANDCENTER).ready:
                    gaz = self.state.vespene_geyser.closer_than(10.0,cc)
                    for vespene in gaz:
                        if not self.can_afford(REFINERY):
                            break
                        worker = self.select_build_worker(vespene.position)
                        if worker is None:
                            break
                        if not self.units(REFINERY).closer_than(1.0, vespene).exists:
                            await self.do(worker.build(REFINERY, vespene))

    async def reset_scout(self):
        if ((self.iteration-self.iteration_scout) / self.iteration_per_min) > 2*2 and len(self.units.of_type(MARINE)) > 0:
            self.iteration_scout=self.iteration
            scv_scout = self.units(SCV)[0]
            #enemy_location = self.known_enemy_structures.random_or(self.enemy_start_locations[0]).position
            n = random.randrange(0,len(self.base_location))
            await self.do(scv_scout.move(self.base_location[n]))


    async def scout(self):
        if self.units(SCV).amount >= 15 and self.scv_scouting < 1:
            self.scv_scouting=1;
            scv_scout = self.units(SCV)[0]
            await self.do(scv_scout.move(self.enemy_start_locations[0]))

    async def expand(self):
        if self.iteration == 1700 or  self.iteration == 2000:
            self.nb_cc += 1
        if self.units.of_type([COMMANDCENTER, ORBITALCOMMAND]).amount < self.nb_cc and self.can_afford(COMMANDCENTER) and self.units(BARRACKS).amount >= 1 :
            await self.expand_now(COMMANDCENTER)

    async def build_offensive_structure(self):
        barracks_placement_position = self.main_base_ramp.barracks_correct_placement
        if self.units.of_type([SUPPLYDEPOT, SUPPLYDEPOTLOWERED, SUPPLYDEPOTDROP]).ready.exists:
            nb_cc = self.units.of_type([COMMANDCENTER, ORBITALCOMMAND]).amount
            if self.units(BARRACKS).amount + self.already_pending(UnitTypeId.BARRACKS) <= 0:
                if self.can_afford(BARRACKS):
                    await self.build(BARRACKS, barracks_placement_position)
            elif (self.units(BARRACKS).amount + self.already_pending(UnitTypeId.BARRACKS) < 4*nb_cc) and self.units(COMMANDCENTER).amount >= 1:
                if self.can_afford(BARRACKS):
                    await self.build(BARRACKS, barracks_placement_position)

            for sp in self.units(BARRACKS).ready.idle:
                if not self.is_in_list(sp.tag) and sp.add_on_tag == 0 and self.can_afford(BARRACKSTECHLAB) and ((self.units(BARRACKSTECHLAB).amount + self.already_pending(UnitTypeId.BARRACKSTECHLAB)) < ((self.units(BARRACKS).amount + self.already_pending(UnitTypeId.BARRACKS))/2)):
                    #await self.do(cc(AbilityId.BUILD_TECHLAB_BARRACKS))
                    self.barrack_try_addon.append(sp.tag)
                    await self.do(sp.build(BARRACKSTECHLAB))

    async def build_offensive_unit(self):
        for barrack in self.units(BARRACKS).ready.noqueue:
            ratio = 100
            if len(self.units.of_type(MARINE)) >0:
                ratio = len(self.units.of_type(MARAUDER)) / len(self.units.of_type(MARINE)) * 100
            if barrack.has_add_on and self.can_afford(MARAUDER) and ratio < 30 and self.supply_left > 0:
                await self.do(barrack.train(MARAUDER))
            if self.can_afford(MARINE) and self.supply_left > 0:
                await self.do(barrack.train(MARINE))

    async def build_research_building(self):
        if not self.units(ENGINEERINGBAY).ready.exists and self.can_afford(ENGINEERINGBAY) and len(self.units(BARRACKS).ready) > 3 and self.already_pending(UnitTypeId.ENGINEERINGBAY) <= 0:
            barracks_placement_position = self.main_base_ramp.barracks_correct_placement
            await self.build(ENGINEERINGBAY, barracks_placement_position)

    async def upgrade(self):
        if self.units(BARRACKSTECHLAB).ready.exists:
            for lab in self.units(BARRACKSTECHLAB).ready:
                abilities = await self.get_available_abilities(lab)
                if AbilityId.RESEARCH_COMBATSHIELD in abilities and self.can_afford(AbilityId.RESEARCH_COMBATSHIELD):
                    await self.do(lab(AbilityId.RESEARCH_COMBATSHIELD))
                elif AbilityId.RESEARCH_CONCUSSIVESHELLS in abilities and self.can_afford(AbilityId.RESEARCH_CONCUSSIVESHELLS):
                    await self.do(lab(AbilityId.RESEARCH_CONCUSSIVESHELLS))
        if self.units(ENGINEERINGBAY).ready.exists:
            for bay in self.units(ENGINEERINGBAY).ready:
                abilities = await self.get_available_abilities(bay)
                if AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL1 in abilities and self.can_afford(AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL1):
                    await self.do(bay(AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL1))
                elif AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL1 in abilities and self.can_afford(AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL1):
                    await self.do(bay(AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL1))


    async def defend(self):
        if self.units.of_type(self.army_units).idle.amount >= 5:
            for struct in self.units.of_type([SUPPLYDEPOT, SUPPLYDEPOTLOWERED, SUPPLYDEPOTDROP,BARRACKS, COMMANDCENTER, REFINERY]).ready:
                for enemy in self.known_enemy_units.not_structure:
                    if enemy.position.to2.distance_to(struct.position.to2) < 10:
                        for ma in self.units.of_type(self.army_units).idle:
                            await self.do(ma.attack(enemy))


    async def attack(self):
        if self.units.of_type(self.army_units).idle.amount > 15:
            choice = random.randrange(0, 4)
            target = False
            count_marine = 0
            count_marauder = 0
            if self.iteration > self.do_something_after:
                if self.use_model:
                    prediction = self.model.predict([self.flipped.reshape([-1,176,200,3])])
                    choice = np.argmax(prediction[0])
                    print('prediction: ',choice)
                else:
                    choice = random.randrange(0, 4)

                if choice == 0:
                    # no attack
                    wait = random.randrange(20, 165)
                    self.do_something_after = self.iteration + wait

                elif choice == 1 and len(self.units.of_type(MARINE).idle) > 35 and len(self.units.of_type(MARAUDER).idle) > 15:
                    #attack_enemy_start
                    count_marine = 60
                    count_marauder = 25
                    if len(self.known_enemy_units) > 0:
                        target = self.enemy_start_locations[0]

                elif choice == 2 and len(self.units.of_type(MARINE).idle) > 15:
                    #attack_closest_structures
                    count_marine = 15
                    if len(self.known_enemy_structures) > 0:
                        target = self.known_enemy_structures.closest_to(random.choice(self.units.of_type([COMMANDCENTER, ORBITALCOMMAND])))

                elif choice == 3 and len(self.units.of_type(MARINE).idle) > 20 and len(self.units.of_type(MARAUDER).idle) > 5:
                    #attack_closest_enemy_units
                    count_marine = 25
                    count_marauder = 5
                    if len(self.known_enemy_units) > 0:
                        target = self.known_enemy_units.closest_to(random.choice(self.units.of_type([COMMANDCENTER, ORBITALCOMMAND])))

                if target:
                    count_order_marine = 0
                    count_order_marauder = 0

                    if choice != 1:
                        target = target.position

                    for army in self.units.of_type(MARINE).idle:
                        if count_order_marine < count_marine:
                            await self.do(army.attack(target))
                            count_order_marine += 1
                        else:
                            break

                    for army in self.units.of_type(MARAUDER).idle:
                        if count_order_marauder < count_marauder:
                            await self.do(army.attack(target))
                            count_order_marauder += 1
                        else:
                            break

                y = np.zeros(4)
                y[choice] = 1
                print(choice)
                self.train_data.append([y,self.flipped])

for i in range(1):
    run_game(maps.get("AbyssalReefLE"), [
        #Human(Race.Terran),
        Bot(Race.Terran, SentdeBot(use_model=False)),
        Computer(Race.Zerg, Difficulty.Hard)
        ], realtime=False)
