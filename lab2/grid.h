typedef struct {
    double u;
    double u1;
    double u2;
} Node;

typedef struct {
    unsigned i;
    unsigned j;
    unsigned rows;
    unsigned cols;
    Node *nodes;
    Node *comNodes;
    char boundaries[4];
} Block;

Block createBlock(unsigned blocks, unsigned process);
