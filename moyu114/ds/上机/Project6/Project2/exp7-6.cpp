#include"btree2.h"
typedef struct {
	int lno;
	BTNode* p;
}qu;
int Nodes(BTNode* b) {
	int n1, n2;
	if (b == NULL)
		return 0;
	else if (b->lchild == NULL && b->rchild == NULL)
		return 1;
	else {
		n1 = Nodes(b->lchild);
		n2= Nodes(b->rchild);
		return (n1 + n2+1);
	}
}
int LeftNodes(BTNode* b) {
	int n1, n2;
	if (b == NULL)
		return 0;
	else if (b->lchild == NULL && b->rchild == NULL)
		return 1;
	else {
		n1 = LeftNodes(b->lchild);
		n2 = LeftNodes(b->rchild);
		return (n1 + n2);
	}
}
int Level(BTNode* b,ElemType x,int h) {
	int l;
	if (b == NULL)
		return (0);
	else if (b->data == x) {
		cout << h<< endl;
		return (h);
	}
		
	else {
		l = Level(b->lchild, x, h + 1);
		if (l != 0)return (1);
		else return(Level(b->rchild, x, h + 1));
	}
}
int Width(BTNode* b) {
	qu Qu[MaxSize];
	int front, rear;
	int lnum, max, i, n;
	front = rear = 0;
	if (b != NULL) {
		rear++;Qu[rear].p = b;
		Qu[rear].lno = 1;
		while (rear != front) {
			front++;b = Qu[front].p;
			lnum = Qu[front].lno;
			if (b->lchild != NULL) {
				rear++;Qu[rear].p = b->lchild;
				Qu[rear].lno = lnum + 1;  //根据左右孩子是否存在记下当前的层次,以便之后扫描队列时，可以分清该层次有多少结点
			}
			if (b->rchild != NULL) {
				rear++;Qu[rear].p = b->rchild;
				Qu[rear].lno = lnum + 1;
			}
		}
			max = 0;lnum = 1;i = 1;
			while (i <= rear) {
				n = 0;
				while (i <= rear && Qu[i].lno == lnum) {
					n++;
					i++;
				}
				lnum = Qu[i].lno;
				(n > max)?max = n:max=max;
			}
			return max;
		}
	
	else
		return 0;
}
int main() {
	ElemType x = 'K';
	BTNode* b, * p, * lp, * rp;
	char str[] = { "A(B(D,E(H(J,K(L,M(,N))))),C(F,G(,I)))" };
	CreateBTNode(b, str);
	cout << "输出二叉树b" << endl;
	DispBTNode(b);
	cout<<endl;
	cout << "（1）二叉树b的结点个数：" << Nodes(b)<<endl;
	cout << "（2）二叉树b的叶子结点个数：" << LeftNodes(b) << endl;
	cout << "（3）二叉树b的结点" << x << "的层次：";
	Level(b, x, 1);
	cout << "（4）二叉树b的宽度：" << Width(b) << endl;
	DestroyBT(b);
}