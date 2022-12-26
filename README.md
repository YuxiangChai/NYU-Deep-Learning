# NYU-Deep-Learning

Course instructed by **Alfredo Canziani** and **Yann LeCun**.

Project part not included, too large. The idea is to use SimCLR as SSL and train the backbone of YoloV5.

---

HW1:  95/100
- Theory: 43
  - 1.2.c, 1.3.a, 1.3.b: incorrect gradients (-2 * 3)
  - 1.4.e: -1
- Implementation: 35 + 15
- Extra credits: 2

HW2:  105/100

HW3:  108.5/100
- 1.1(a) -0.5 Missing key concept of "low energy"
- 1.1(b) -1 Missing the key idea of p(y|x) and F_w(x,y)
- 1.2(d) -1 Key concept of NLL pushes the energy up with a force not proportional to the distance from the correct y, but proportional to the probability of that particular y' is paritally explained.
- 1.3(d)(iii) -1 In the situation how these losses are impacted by outliers is missing.

Implementation
-3 Worst paths need to start at top left and end at bottom right

15+ GTN Extra Credit
