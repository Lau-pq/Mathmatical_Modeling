d e f c r o s s o v e r _ a n d _ m u t a t i o n ( pop , CROSSOVER_RATE = 0 . 5 ) :
f o r i i n range ( l e n ( pop ) ) :
i f np . random . r a n d ( ) < CROSSOVER_RATE:
m ot h e r _i d = np . random . r a n d i n t ( POP_SIZE )