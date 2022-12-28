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
				Qu[rear].lno = lnum + 1;  //�������Һ����Ƿ���ڼ��µ�ǰ�Ĳ��,�Ա�֮��ɨ�����ʱ�����Է���ò���ж��ٽ��
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
	cout << "���������b" << endl;
	DispBTNode(b);
	cout<<endl;
	cout << "��1��������b�Ľ�������" << Nodes(b)<<endl;
	cout << "��2��������b��Ҷ�ӽ�������" << LeftNodes(b) << endl;
	cout << "��3��������b�Ľ��" << x << "�Ĳ�Σ�";
	Level(b, x, 1);
	cout << "��4��������b�Ŀ�ȣ�" << Width(b) << endl;
	DestroyBT(b);
}