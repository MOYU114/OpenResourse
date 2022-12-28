#include"btree3.h"
#include<stdlib.h>
#include<string.h>
BTNode* CRTree(char s[], int i, int j) {
	BTNode* p;
	int k, plus = 0, posi;
	if (i == j) {
		p = (BTNode*)malloc(sizeof(BTNode));
		p->data = s[i];
		p->lchild = p->rchild = NULL;
		return p;
	}
	for(k=i;k<=j;k++)
		if (s[k] == '+' || s[k] == '-'){
			plus++;
			posi=k;
		}
	if(plus==0)
		for(k=i;k<=j;k++)
			if (s[k] == '*' || s[k] == '/') {
				plus++;
				posi = k;
			}
	if(plus!=0){
		p = (BTNode*)malloc(sizeof(BTNode));
		p->data = s[posi];
		p->lchild = CRTree(s, i, posi - 1);
		p->rchild = CRTree(s, posi + 1,j);
		return p;
	}
	else {
		return NULL;
	}
}
double compute(BTNode* b) {
	double v1, v2;
	if (b == NULL)return 0;
	if (b->lchild == NULL && b->rchild == NULL)
		return b->data - '0';
	v1 = compute(b->lchild);
	v2 = compute(b->rchild);
	switch (b->data) {
	case '+':
			return v1 + v2;
	case '-':
		return v1 - v2;
	case '*':
		return v1 * v2;
	case '/':
		if(v2!=0)
			return v1 / v2;
		else {
			cout << "计算有误";
			abort();
		}
		
	}
}
int main() {
	BTNode* b;
	char s[MaxSize] = { "1+2*3-4/5" };
	cin >> s;
	b = CRTree(s, 0, strlen(s) - 1);
	cout << "对应二叉树：";
	DispBTNode(b);
	cout << endl;
	cout << "计算结果为：" << compute(b);
	DestroyBT(b);
}