#include<iostream>
#include<malloc.h>
using namespace std;
typedef int KeyType;
typedef char InfoData;
//KeyType predt = -32767;
typedef struct Node {
	KeyType key;
	InfoData data;
	struct Node* lchild, * rchild;
}BSTNode;
bool InsertBST(BSTNode*& bt, KeyType k);
BSTNode* CreateBST(KeyType a[], int n);
void DeleteChild(BSTNode* p, BSTNode*& r);
void Delete(BSTNode*& p);
bool DeleteBST(BSTNode*& bt, KeyType k);
void SearchBST1(BSTNode* bt, KeyType k, KeyType path[], int i);
int SearchBST2(BSTNode* bt, KeyType k);
void DispBST(BSTNode* bt);

bool JudgeBST(BSTNode* bt);
void DestoryBST(BSTNode* bt);