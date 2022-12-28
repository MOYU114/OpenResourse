#include"bst.h"
bool InsertBST(BSTNode*& bt, KeyType k) {
	if (bt == NULL) {
		bt = (BSTNode*)malloc(sizeof(BSTNode));
		bt->lchild = bt->rchild = NULL;
		bt->key = k;
		return true;
	}
	else if(k==bt->key){
		return false;
	}
	else if (k <bt->key) {
		InsertBST(bt->lchild, k);
	}
	else if (k > bt->key) {
		InsertBST(bt->rchild, k);
	}
}
void DispBST(BSTNode *bt);
BSTNode* CreateBST(KeyType a[], int n) {
	BSTNode* bt = NULL;
	int i = 0;
	while (i < n) {
		if (InsertBST(bt, a[i]) == true) {
			printf("第%d步，插入%d：", i + 1, a[i]);
			DispBST(bt);
			cout << endl;
			i++;
		}
		
	}return bt;
}
void DeleteChild(BSTNode* p, BSTNode*& r) {//删除p后处理其左右孩子，r为左孩子
	BSTNode* q;
	if (r->rchild != NULL)
		DeleteChild(p, r->rchild);
	else {
		p->key = r->key;
		p->data = r->data;
		q = r;
		r = r->lchild;
		free(q);
	}
}
void Delete(BSTNode*& p) {//删除单个结点
	BSTNode *q;
	if (p->rchild == NULL) {
		q = p;p = p->lchild;free(q);
	}
	else if (p->lchild == NULL) {
		q = p;p = p->rchild;free(q);
	}
	else
		DeleteChild(p, p->lchild);
}
bool DeleteBST(BSTNode*& bt, KeyType k) {
	if (bt == NULL) return false;
	else {
		if (k < bt->key) {
			return DeleteBST(bt->lchild, k);
		}else if (k > bt->key) {
			return DeleteBST(bt->rchild, k);
		}
		else {//k == bt->key
			Delete(bt);
			return true;
		}
	}
}
void SearchBST1(BSTNode* bt, KeyType k, KeyType path[], int i) {
	int j;
	if(bt == NULL)
		return;
	else if (k == bt->key) {
		path[i + 1] = bt->key;
		for (j = 0;j <= i + 1;j++)
			printf("%3d", path[j]);
		cout << endl;
	}
	else {
		path[i + 1] = bt->key;
		if (k < bt->key) 
			SearchBST1(bt->lchild, k, path, i + 1);
		else 
			SearchBST1(bt->rchild, k, path, i + 1);
	}
}
int SearchBST2(BSTNode* bt, KeyType k) {
	if (bt == NULL)
		return 0;
	else if (k == bt->key) {
		printf("%3d", bt->key);
		return 1;
	}if (k < bt->key)
		SearchBST2(bt->lchild, k);
	else
		SearchBST2(bt->rchild, k);
	printf("%3d", bt->key);
}
void DispBST(BSTNode* bt) {
	if (bt != NULL) {
		printf("%d", bt ->key);
		if (bt->lchild != NULL || bt->rchild != NULL) {
			cout << "(";
			DispBST(bt->lchild);
			if (bt->rchild != NULL)cout << ",";
			DispBST(bt->rchild);
			cout << ")";
		}
	}
}
KeyType predt = -32767;
bool JudgeBST(BSTNode* bt) {
	bool b1, b2;
	if (bt == NULL)
		return true;
	else {
		b1 = JudgeBST(bt->lchild);
		if (b1 == false || predt >= bt->key)//判断前驱结点值与当前结点值的大小
			return false;
		predt = bt->key;
		b2 = JudgeBST(bt->rchild);
		return b2;
	}
}
void DestoryBST(BSTNode* bt) {
	if (bt != NULL) {
		DestoryBST(bt->lchild);
		DestoryBST(bt->rchild);
		free(bt);
	}
}