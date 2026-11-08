"""The built-in training text for the from-scratch character model.

Original prose written for this project, lowercase, a small alphabet, about 8 KB. Two parts:

* stories, for open-ended prompts, full of the common phrases ("the sea", "in the morning")
  that greedy decoding falls into and cannot climb out of;
* facts, for single-answer prompts ("the capital of france is ..."), each with exactly one
  right completion, so determinism can be scored rather than eyeballed.
"""

from __future__ import annotations

STORIES = """\
the old lighthouse stood at the edge of the sea. every night the keeper climbed the stairs and \
lit the lamp, and every morning he walked down to the harbour to watch the boats come in. the \
sea was grey in the morning and green in the afternoon, and at night it was black and full of \
stars.
the keeper had a small dog and a large garden. in the garden he grew beans and potatoes and \
tall yellow flowers that turned to follow the sun. the dog slept by the door and barked at the \
gulls, and the gulls laughed at the dog from the roof.
one winter a storm came in from the west. the wind pulled at the roof and the rain beat against \
the windows. the keeper sat by the fire and listened to the sea. in the morning the beach was \
covered with wood and rope and the bones of an old boat. the dog ran along the sand and the \
keeper walked behind him, picking up the rope.
in the spring a girl came to the village to paint the boats. she painted the red boat and the \
blue boat and the small white boat that belonged to the baker. she painted the harbour in the \
morning and the harbour at night. the keeper asked her to paint the lighthouse, and she said \
she would paint it when the light was right.
the baker woke before the sun. he made bread for the village and cakes for the market, and he \
sold fish on the days when his boat came home full. the children of the village waited by his \
window in the morning, and he gave them the small round loaves that came out of the oven last.
the market was held on the square by the church. there were apples and onions, honey and \
cheese, and an old man who sold clocks that did not tell the time. people came from the hills \
to buy salt and rope, and the fishermen sold their fish from wooden boxes full of ice.
a fox lived in the hills above the village. at night he came down to the gardens to look for \
eggs, and in the morning he went back up the hill with his tail held high. the keeper saw him \
from the top of the lighthouse and did not tell anyone, because he liked the fox.
the river ran down from the hills to the sea. in the summer it was slow and warm, and the \
children swam in the deep pool under the bridge. in the winter it was fast and cold, and it \
carried stones and branches down to the harbour.
when the girl finished her painting of the lighthouse she gave it to the keeper. he hung it \
by the window at the top of the stairs, where he could see the painting and the sea at the \
same time. the painted sea was calm and the real sea was not, and the keeper liked them both.
the years went by. the keeper grew old and the dog grew slow. the girl came back every spring \
to paint the boats, and the baker made bread, and the fox came down from the hills at night. \
the light turned and turned above the sea, and the ships went safely home.
on a quiet evening the keeper climbed the stairs for the last time. he lit the lamp and looked \
out over the water. the sea was calm and the sky was full of stars. far away a small boat was \
coming home, and the light showed it the way.
"""

FACTS = """\
the capital of france is paris.
the capital of italy is rome.
the capital of spain is madrid.
the capital of japan is tokyo.
the capital of egypt is cairo.
the capital of norway is oslo.
the capital of greece is athens.
the capital of peru is lima.
the colour of the sky is blue.
the colour of grass is green.
the colour of snow is white.
the colour of coal is black.
the opposite of hot is cold.
the opposite of up is down.
the opposite of day is night.
the opposite of old is young.
the opposite of fast is slow.
two plus two is four.
three plus four is seven.
five plus five is ten.
six plus three is nine.
the french word for cat is chat.
the french word for dog is chien.
the french word for sea is mer.
the french word for house is maison.
the spanish word for water is agua.
the german word for book is buch.
"""

# Facts appear twice so the tiny model memorises them; the stories carry the open-ended text.
TINY_CORPUS = STORIES + "\n" + FACTS + "\n" + STORIES + "\n" + FACTS
