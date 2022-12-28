#include<iostream>
#include<malloc.h>
using namespace std;
#include"btree.h"
#define MAXWidth 40
BTNode* CreateBT1(char* pre, char* in, int n) {
	BTNode* b;
	char* p;
	int k;
	if (n <= 0) return NULL;//空树或不存在
	b = (BTNode*)malloc(sizeof(BTNode));
	b->data = *pre;
	for (p = in;p < in + n;p++)//在中序中寻找与*p元素相等的元素
		if (*p == *pre)//找到匹配元素，即为根结点
			break;
	k = p - in;//k为根结点其在中序中的位置
	b->lchild = CreateBT1(pre + 1, in, k);//递归调用查询其左右孩子
	b->rchild = CreateBT1(pre + 1+k, p+1,n-1-k);
	return b;
}
BTNode* CreateBT2(char* post, char* in, int n) {
	BTNode* b;
	char r,*p;
	int k;
	if (n <= 0) return NULL;//空树或不存在
	b = (BTNode*)malloc(sizeof(BTNode));
	r = *(post + n - 1);
	b->data = r;
	for (p = in;p < in + n;p++)//在中序中寻找与*p元素相等的元素
		if (*p == r)//找到匹配元素，即为根结点
			break;
	k = p - in;//k为根结点其在中序中的位置
	b->lchild = CreateBT2(post, in, k);//递归调用查询其左右孩子
	b->rchild = CreateBT2(post + k, p + 1, n - 1 - k);
	return b;
}
//括号表示法为btree库中的DispBTNode();
void DispBTree1(BTNode*b) {//凹凸法
	BTNode* St[MAXWidth], * p;
	int level[MAXWidth][2], top = -1, n, i, width = 4;//初始化 level[][0]存储结点宽度，level[][1]存储结点属性
	char type;        //左右孩子标记 
	if (b != NULL) {
		top++;St[top] = b;  //根进栈
		level[top][0] = width;//记录结点属性
		level[top][1] = 2;  
		while (top > -1) {
			p = St[top];//栈顶结点出栈
			n = level[top][0];//读取其宽度
			switch (level[top][1]) {//将其属性赋予type
			case 0:type = 'L';break;
			case 1:type = 'R';break;
			case 2:type = 'B';break;
			}
			for (i = 1;i <= n;i++)
				cout << " ";
			printf("%c(%c)", p->data, type);
			for (i = n + 1;i <= MAXWidth;i += 2)
				cout << "--";
			cout << endl;
			top--;
			if (p->rchild != NULL) {
				top++;
				St[top] = p->rchild;
				level[top][0] = n + width;//宽度增加，表示不与当前结点同层
				level[top][1] = 1;
			}
			if (p->lchild != NULL) {
				top++;
				St[top] = p->lchild;
				level[top][0] = n + width;//宽度增加，表示不与当前结点同层
				level[top][1] = 0;
			}
		}

	}
}
int main() {
	BTNode* b;
	ElemType pre[] = "ABDEHJKLMNCFGI";
	ElemType in[] = "DBJHLKMNEAFCGI";
	ElemType post[] = "DJLNMKHEBFIGCA";
	int n = 14;
	b = CreateBT1(pre, in, n);
	printf("先序遍历：%s\n", pre);
	printf("中序遍历：%s\n", in);
	printf("该树的括号表示法：");
	DispBTNode(b);cout << endl;
	printf("该树的凹凸表示法：\n");
	DispBTree1(b);cout << endl;

	b = CreateBT2(post, in, n);
	printf("中序遍历：%s\n", in);
	printf("后序遍历：%s\n", post);
	printf("该树的括号表示法：");
	DispBTNode(b);cout << endl;
	printf("该树的凹凸表示法：\n");
	DispBTree1(b);cout << endl;
	DestroyBT(b);
}