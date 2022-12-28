#include<iostream>
#include<malloc.h>
using namespace std;
#define MaxSize 50
typedef char ElemType;
typedef struct node
{
    ElemType data;
    struct node* lchild, * rchild;
}   BTNode;
void CreateBTNode(BTNode*& b, char* str);
void DestroyBT(BTNode*& b);
BTNode* FindNode(BTNode* b, ElemType x);
BTNode* LchildNode(BTNode* p);
BTNode* RchildNode(BTNode* p);
int BTNodeDepth(BTNode* b);
void DispBTNode(BTNode* b);