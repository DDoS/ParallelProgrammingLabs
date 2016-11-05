typedef struct {
    unsigned rows;
    unsigned cols;
} Partition;

typedef struct {
    float u;
    float u1;
    float u2;
} Node;

typedef struct {
    unsigned index;
    unsigned i;
    unsigned j;
    unsigned rows;
    unsigned cols;
    Node *nodes;
    Node *aboveNodes;
    Node *rightNodes;
    Node *belowNodes;
    Node *leftNodes;
} Block;

Partition createPartition(unsigned processCount);

Block createBlock(Partition *partition, unsigned process);

void updateBlock(Partition *partition, Block *block);
