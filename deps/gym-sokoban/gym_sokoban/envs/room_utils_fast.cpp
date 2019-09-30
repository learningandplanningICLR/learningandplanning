#include <stdio.h>
#include <iostream>
#include <cstring>
#include <random>
#include <unordered_set>
#include <stdint.h>

// max allocation size
const int max_size = 20;
const int max_boxes = 10;
// room parameters
int num_boxes;
int dim[2];
// 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
int CHANGE_COORDINATES[4][2] = {{-1, 0}, {1, 0},  {0, -1}, {0, 1}};

// https://www.boost.org/doc/libs/1_69_0/boost/container_hash/hash.hpp
// https://stackoverflow.com/questions/17016175/c-unordered-map-using-a-custom-class-type-as-the-key
// struct for 2d array for hash purpose (need operator==)
struct Key {
  uint8_t arr[max_size][max_size];
  uint8_t* operator[](int i) { return arr[i]; }
  bool operator==(const Key &other) const { 
    for (int i = 0; i < dim[0]; ++i) {
      for (int j = 0; j < dim[1]; ++j) {
        if (arr[i][j] != other.arr[i][j])
          return false;
      }
    }
    return true;
  }
};

// hashing function for unordered_set and Key
struct KeyHasher {
  std::size_t operator()(const Key& k) const
  {
    int seed = 0;
    for (int i = 0; i < dim[0]; ++i) {
      for (int j = 0; j < dim[1]; ++j) {
        seed ^= k.arr[i][j] + 0x9e3779b9 + (seed<<6) + (seed>>2);
      }
    }
    return seed;
  }
};

// random number generator 
std::mt19937 rng;

// DFS bookkeeping
int best_room_score = -1;
Key best_room;
// nth box target = [n][0], nth box location [n][1]
uint8_t best_box_mapping[max_boxes][2][2];  
std::unordered_set<Key, KeyHasher> explored_states;

/*
  Calculates the sum of all Manhattan distances, between the boxes
  and their origin box targets.
  :param box_mapping:
  :return:
*/
int box_displacement_score(uint8_t box_mapping[max_boxes][2][2]) {
  int score = 0;
  int dist = 0; 
  for (int i = 0; i < num_boxes; ++i) {
    dist = abs(box_mapping[i][0][0]-box_mapping[i][1][0])
         + abs(box_mapping[i][0][1]-box_mapping[i][1][1]);
    score += dist;
  }
  return score;
}

/*
  Searches through all possible states of the room.
  This is a recursive function, which stops if the tll is reduced to 0 or
  over 300000 states have been explored.
  :param room_state:                            
  :param room_structure:
  :param box_mapping:
  :param box_swaps:
  :param last_pull:
  :param ttl:
  :return:
*/

void depth_first_search(Key &room_state, 
                        Key &room_structure, 
                        uint8_t player_position[2],
                        uint8_t box_mapping[max_boxes][2][2],
                        Key &box_to_target,
                        int empty_box_targets,
			                  int box_swaps,
			                  int last_pull,
			                  int ttl) {

  --ttl;
  if (ttl <= 0 || explored_states.size() >= 300000)
    return;

  // Only search this state, if it not yet has been explored
  if (explored_states.find(room_state) == explored_states.end()) {
    // Add current state and its score to explored states
    int room_score = box_swaps * box_displacement_score(box_mapping);

    if (empty_box_targets != num_boxes) 
      room_score = 0;

    if (room_score > best_room_score) {
      // update best_room_score
      best_room_score = room_score;
      // update best_room
      for (int i = 0; i < dim[0]; ++i) {
        memcpy(&best_room.arr[i], &room_state.arr[i], sizeof(uint8_t)*dim[1]);
      }
      // update best_box_mapping
      for (int i = 0; i < num_boxes; ++i) {
        for (int j = 0; j < 2; ++j) {
          for (int k = 0; k < 2; ++k) {
            best_box_mapping[i][j][k] = box_mapping[i][j][k];
          }
        }
      }
    }

    // initialize variables before recursive calls
    explored_states.insert(room_state);
    uint8_t next_player_position[2];
    uint8_t possible_box_location[2];
    uint8_t change[2];
    bool moved_box = false;
    int new_status;  // new status of moved box
    int idx = -1;

    Key next_room_state;
    uint8_t next_box_mapping[max_boxes][2][2];
    Key next_box_to_target;

    // perform recursive call for each action (actions < 4 move box; actions >=4 move player only):
    // 0: 'push up',
    // 1: 'push down',
    // 2: 'push left',
    // 3: 'push right',
    // 4: 'move up',
    // 5: 'move down',
    // 6: 'move left',
    // 7: 'move right',
    for (int action = 0; action < 8; ++action) {
      // create a copy of room_state;
      for (int i = 0; i < dim[0]; ++i) {
        memcpy(&next_room_state.arr[i], &room_state.arr[i], sizeof(uint8_t)*dim[1]);
      }
      // copy box_to_target
      for (int i = 0; i < dim[0]; ++i) {
        memcpy(&next_box_to_target.arr[i], &box_to_target.arr[i], sizeof(uint8_t)*dim[1]);
      }
      // copy box_mapping
      for (int i = 0; i < num_boxes; ++i) {
        for (int j = 0; j < 2; ++j) {
          for (int k = 0; k < 2; ++k) {
            next_box_mapping[i][j][k] = box_mapping[i][j][k];
          }
        }
      }
      moved_box = false;
      idx = -1;
      // DO the 'reverse move'
      change[0] = CHANGE_COORDINATES[action % 4][0];
      change[1] = CHANGE_COORDINATES[action % 4][1];
      next_player_position[0] = player_position[0] + change[0];
      next_player_position[1] = player_position[1] + change[1];
      // Check if next position is an empty floor or an empty box target
      if (room_state[next_player_position[0]][next_player_position[1]] == 1 ||
          room_state[next_player_position[0]][next_player_position[1]] == 2) {
        // Move player, independent of pull or move action.
        next_room_state[player_position[0]][player_position[1]] = 
          room_structure[player_position[0]][player_position[1]];
        if (room_state[next_player_position[0]][next_player_position[1]] == 1)
          next_room_state[next_player_position[0]][next_player_position[1]] = 5;
        else
          next_room_state[next_player_position[0]][next_player_position[1]] = 6;
        // In addition try to pull a box if the action is a pull action
        if (action < 4) {
          possible_box_location[0] = player_position[0]-change[0];
          possible_box_location[1] = player_position[1]-change[1];
          new_status = next_room_state[possible_box_location[0]][possible_box_location[1]];
          // check if there is a box on possbile_box_location
          if (new_status == 4 || new_status == 3) {
            moved_box = true;
            // moved from non-target cell to target cell 
            if (room_structure[possible_box_location[0]][possible_box_location[1]] == 1 && room_structure[player_position[0]][player_position[1]] == 2) {
              --empty_box_targets;
              new_status = 3;
            }
            // moved from target cell to non-target 
            if (room_structure[possible_box_location[0]][possible_box_location[1]] == 2 && room_structure[player_position[0]][player_position[1]] == 1) {
              ++empty_box_targets;
              new_status = 4;
            }
            // Perform pull of the adjacent box
            next_room_state[player_position[0]][player_position[1]] = new_status;
            next_room_state[possible_box_location[0]][possible_box_location[1]] 
              = room_structure[possible_box_location[0]][possible_box_location[1]];
            // index of moved box
            idx = next_box_to_target[possible_box_location[0]][possible_box_location[1]];
            // update box_to_target; 
            next_box_to_target[player_position[0]][player_position[1]] = idx;
            // update box_mapping
            next_box_mapping[idx][1][0] = player_position[0];
            next_box_mapping[idx][1][1] = player_position[1];
          }
        }
      }
      if (idx != last_pull) {
        ++box_swaps;
      }

      // Perform the recursive call
      depth_first_search(next_room_state,  // gets modified
                         room_structure, 
                         next_player_position,
                         next_box_mapping,  // gets modified
                         next_box_to_target,  // gets modified
                         empty_box_targets,
                         box_swaps,
                         idx, 
                         ttl); 

      // UNDO the 'reverse move'
      if (idx != last_pull) {
        --box_swaps;
      }
      if (moved_box) {
        // moved from non-target cell to target cell 
        if (room_structure[possible_box_location[0]][possible_box_location[1]] == 1 && room_structure[player_position[0]][player_position[1]] == 2) {
          ++empty_box_targets;
        }
        // moved from target cell to non-target 
        if (room_structure[possible_box_location[0]][possible_box_location[1]] == 2 && room_structure[player_position[0]][player_position[1]] == 1) {
          --empty_box_targets;
        }
      }
    }
  }
}

/*
  This function plays Sokoban reverse in a way, such that the player can
  move and pull boxes.
  It ensures a solvable level with all boxes not being placed on a box target.
  :param room_state:
  :param room_structure:
  :param curriculum controls the depth of the DFS in reverse_playing:
  :return best score:
*/
int reverse_playing(Key &room_state,
                    Key &room_structure, 
                    uint8_t box_mapping[max_boxes][2][2],
                    int curriculum) {

  // box_to_target maps a position of a box to index of a target
  Key box_to_target;
  // player position
  uint8_t player_position[2] = {0, 0};
  // we start with box on targets
  int idx = 0;
  for (int i = 0; i < dim[0]; ++i) {
    for (int j = 0; j < dim[1]; ++j) {
      if (room_structure[i][j] == 2) {
        box_mapping[idx][0][0] = i;
        box_mapping[idx][0][1] = j;
        box_mapping[idx][1][0] = i;
        box_mapping[idx][1][1] = j;
        box_to_target[i][j] = idx;
        best_box_mapping[idx][0][0] = i;
        best_box_mapping[idx][0][1] = j;
        best_box_mapping[idx][1][0] = i;
        best_box_mapping[idx][1][1] = j;
        ++idx;
      }
      if (room_state[i][j] == 5) {
        player_position[0] = i;
        player_position[1] = j;
      }
    }
  }

  explored_states.clear();
  best_room_score = -1;
  int last_pull = -1; 

  depth_first_search(room_state, 
                     room_structure, 
                     player_position,
                     box_mapping, 
                     box_to_target,
                     0,  // empty_box_targets=0, since initially boxes on targets
                     0,  // box_swaps
                     last_pull, 
                     curriculum);  // ttl

  return best_room_score;
}

/*
  Places the player and the boxes into the floors in a room.
  :param room:
  :param num_boxes:
  :return error code:
  -3: "Not enough free spots (#%d) to place %d player and %d boxes.\n"
  0: OK
*/
int place_boxes_and_player(Key &room,
                           bool second_player) {
  // Get all available positions
  int num_possible_positions = 0;
  for (int i = 0; i < dim[0]; ++i) {
    for (int j = 0; j < dim[1]; ++j) {
      if (room[i][j] == 1) 
        ++num_possible_positions;
    }
  }
  uint8_t possible_positions[num_possible_positions][2];
  int idx = 0;
  for (int i = 0; i < dim[0]; ++i) {
    for (int j = 0; j < dim[1]; ++j) {
      if (room[i][j] == 1) {
        possible_positions[idx][0] = i;
        possible_positions[idx][1] = j;
        ++idx;
      }
    }
  }
  int num_players = (second_player) ? 2 : 1;

  if (num_possible_positions <= num_boxes + num_players) {
    //printf("Not enough free spots (#%d) to place %d player and %d boxes.\n",
    //  num_possible_positions, num_players, num_boxes);
    return -3;
  }

  // Place player(s)
  std::uniform_int_distribution<> dist(0, num_possible_positions-1);
  idx = dist(rng);
  // possible_positions is (# of 1's) x 2
  int player_position[2] = {possible_positions[idx][0], 
                             possible_positions[idx][1]};
  room[player_position[0]][player_position[1]] = 5;
  --num_possible_positions;

  if (second_player) {
    idx = dist(rng);
    player_position[0] = possible_positions[idx][0];
    player_position[1] = possible_positions[idx][1];
    room[player_position[0]][player_position[1]] = 5;
    --num_possible_positions;
  }

  // Place boxes
  for (int n = 0; n < num_boxes; ++n) {
    // find possible positions again
    std::uniform_int_distribution<> dist_box(0, num_possible_positions-1);
    int possible_positions_new[num_possible_positions][2];
    idx = 0;
    for (int i = 1; i < dim[0]-1; ++i) {
      for (int j = 1; j < dim[1]-1; ++j) {
        if (room[i][j] == 1) {
          possible_positions_new[idx][0] = i;
          possible_positions_new[idx][1] = j;
          ++idx;
        }
      }
    }
    idx = dist_box(rng);
    room[possible_positions_new[idx][0]]
        [possible_positions_new[idx][1]] = 2;
    --num_possible_positions;
  }
  return 0;
}

/*
   Generate a room topology, which consits of empty floors and walls.

   :param dim:
   :param p_change_directions:
   :param num_steps:
   :return:
   */
void room_topology_generation(Key &room, 
                              double p_change_directions, 
                              int num_steps) {
  // The ones in the mask represent all fields which will be set to floors
  // during the random walk. The centered one will be placed over the current
  // position of the walk.
  uint8_t masks[5][3][3] = {
    {
      {0, 0, 0},
      {1, 1, 1},
      {0, 0, 0}
    },
    {
      {0, 1, 0},
      {0, 1, 0},
      {0, 1, 0}
    },
    {
      {0, 0, 0},
      {1, 1, 0},
      {0, 1, 0}
    },
    {
      {0, 0, 0},
      {1, 1, 0},
      {1, 1, 0}
    },
    {
      {0, 0, 0},
      {0, 1, 1},
      {0, 1, 0}
    }
  };

  // Possible directions during the walk
  std::uniform_int_distribution<> dist_a(0, 3);
  std::uniform_int_distribution<> dist_x(1, dim[0]-2);
  std::uniform_int_distribution<> dist_y(1, dim[1]-2);
  std::uniform_int_distribution<> dist_m(0, 4);
  std::uniform_int_distribution<> dist_d(1, 100);
  int idx = dist_a(rng);
  int direction[2] = {CHANGE_COORDINATES[idx][0], 
                      CHANGE_COORDINATES[idx][1]};

  // Starting position of random walk
  int position[2] = {dist_x(rng), dist_y(rng)};

  for (int s = 0; s < num_steps; ++s) {
    // Change direction randomly 
    if (dist_d(rng) < 100*p_change_directions) {
      idx = dist_a(rng);
      direction[0] = CHANGE_COORDINATES[idx][0]; 
      direction[1] = CHANGE_COORDINATES[idx][1]; 
    }

    // Update position
    position[0] += direction[0];  
    position[1] += direction[1];
    position[0] = std::max(std::min(position[0], dim[0] - 2), 1);
    position[1] = std::max(std::min(position[1], dim[1] - 2), 1);

    // Apply mask 
    int choose_mask = dist_m(rng);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        room[i+position[0]-1][j+position[1]-1] += masks[choose_mask][i][j];
      }
    }
  }
  for (int i = 0; i < dim[0]; ++i) {
    room[i][0] = 0;
    room[i][dim[1]-1] = 0;
    for (int j = 0; j < dim[1]; ++j) {
      if (room[i][j] > 0) 
        room[i][j] = 1;
    }
  }
  for (int j = 0; j < dim[1]; ++j) {
    room[0][j] = 0;
    room[dim[0]-1][j] = 0;
  }
}

/*
  Generates a Sokoban room, represented by an integer matrix. The elements are encoded as follows:
  0: wall
  1: empty space
  2: box target
  3: box on target
  4: box off target
  5: player

  :param dim:
  :param p_change_directions:
  :param num_steps:
  :return error codes:
  -3: BAD, "Not enough free spots (#%d) to place %d player and %d boxes." (place_box_and_players)
  -2: BAD, "More boxes (%d) than allowed (%d)!"
  -1: BAD, "Sokoban size (%dx%d) bigger than maximu size (%dx%d)!"
  else: best score. 
*/
int generate_room_c(int d[2],
                    double p_change_directions,
                    int num_steps,
                    int nb,  // num boxes
                    int tries,
                    unsigned int seed,
                    bool do_reverse_playing,
                    bool second_player,
                    int curriculum) {

  // assert
  if (d[0] > max_size || d[1] > max_size) {
    //printf("Sokoban size (%dx%d) bigger than maximu size (%dx%d)!", d[0], d[1], max_size, max_size);
    return -1;
  }
  if (nb > max_boxes) {
    //printf("More boxes (%d) than allowed (%d)!", nb, max_boxes);
    return -2;
  }

  // intialize global variables
  dim[0] = d[0];
  dim[1] = d[1];
  num_boxes = nb;
  rng.seed(seed);

  // room is populated randomly and then copied to room_structure and room_state
  Key room;
  // Room fixed represents all not movable parts of the room
  // Room structure represents the current state of the room including movable parts
  Key room_state;
  Key room_structure;

  // box_mapping maps box_target to a corrsponding box location
  uint8_t box_mapping[max_boxes][2][2];
  int score = 0;

  // Sometimes rooms with a score == 0 are the only possibility.
  // In these case, we try another model.
  for (int t = 0; t < tries; ++t) { 
    // reset room
    for (int i = 0; i < dim[0]; ++i) {
      memset(&room.arr[i], 0, sizeof(uint8_t)*dim[1]);
    }
    // sets entries to 0 or 1
    room_topology_generation(room, p_change_directions, num_steps);
    // populates some allowed entries (1) with 2 (goal) or 5 (player)
    score = place_boxes_and_player(room, second_player);

    for (int i = 0; i < dim[0]; ++i) {
      for (int j = 0; j < dim[1]; ++j) {
        room_state[i][j] = room[i][j];
        room_structure[i][j] = room[i][j];
        if (room[i][j] == 5) 
          room_structure[i][j] = 1;
        if (room[i][j] == 2) 
          room_state[i][j] = 3;  // initially all boxes on target
      }
    }

    // TODO: Here we want to control the number of reverse actions
    if (do_reverse_playing)
      score = reverse_playing(room_state, room_structure, box_mapping, curriculum);

    if (score > 0) break;
  }

  //if (score == 0) {
  //  printf("Generated Model with score == 0");
  //  exit(0);
  //}
  return score;
}

extern "C" {
int generate_room(uint8_t *result, //result[max_size*max_size],
                  int d[2], 
                  double p_change_directions, 
                  int num_steps, 
                  int nb,  // num boxes
                  int tries, 
                  unsigned int &seed,
                  bool do_reverse_playing,
                  bool second_player,
                  int curriculum) {  // curriculum controls the depth of DFS in reverse_playing

  int score = generate_room_c(d,
                              p_change_directions,  // p_change_direction
                              num_steps,  // num_steps
                              nb,  // num boxes
                              tries,  
                              seed,
                              do_reverse_playing,
                              second_player,
                              curriculum);

  // sample next uint32 seed
  std::uniform_int_distribution<unsigned int> dist_s(0, 0xFFFFFFFF);
  seed = dist_s(rng);

  for (int i = 0; i < dim[0]; ++i) {
    for (int j = 0; j < dim[1]; ++j) {
      result[i*dim[0]+j] = best_room[i][j];
    }
  }

  return score;
}
}

int main(int argc, char** argv) {
  // initialize global variables
  int d[2] = {8, 8};
  int nb = 2;
  int tries = 4;
  std::random_device rd;
  unsigned int seed;
  int curriculum = 300;
  if (argc == 1)
    seed = rd();
  else if (argc == 2)
    seed = atoi(argv[1]);  // 76878694
  else if (argc == 3) {
    seed = atoi(argv[1]);  // 76878694
    curriculum = atoi(argv[2]);  
  }
  else
    printf("Too many arguments.\n");

  printf("Seed: %u\n", seed);

  uint8_t res[d[0]*d[1]];
  int num_steps = (int)(1.7 * (d[0] + d[1]));
  printf("num steps: %d\n", num_steps);

  //int score = generate_room(res,
  //            d,
  //            0.35,  // p_change_direction
  //            num_steps,  // num_steps
  //            nb,  // num boxes
  //            tries,
  //            seed,
  //            true,  // do_reverse_playing
  //            0,
  //            curriculum);  // 
  //
  //for (int i = 0; i < dim[0]; ++i) {
  //  for (int j = 0; j < dim[1]; ++j) {
  //    printf("%d", res[i*dim[0]+j]);
  //  }
  //  printf("\n");
  //}

  //printf("Score: %d. Next seed: %u\n", score, seed);


  std::uniform_int_distribution<> dist_c(10, 300);
  for (int i = 0; i < 1000000; ++i) {
    int tmp = seed;
    int curriculum = dist_c(rd);
    int score = generate_room(res,
                d,
                0.35,  // p_change_direction
                num_steps,  // num_steps
                nb,  // num boxes
                tries,
                seed,
                true,  // do_reverse_playing
                false,
                curriculum);
    if (score < 0) 
      printf("%d, %u\n", score, tmp);
  }
}


