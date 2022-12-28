#include<iostream>
#include<malloc.h>
using namespace std;
#include"btree.h"
#define MAXWidth 40
BTNode* CreateBT1(char* pre, char* in, int n) {
	BTNode* b;
	char* p;
	int k;
	if (n <= 0) return NULL;//�����򲻴���
	b = (BTNode*)malloc(sizeof(BTNode));
	b->data = *pre;
	for (p = in;p < in + n;p++)//��������Ѱ����*pԪ����ȵ�Ԫ��
		if (*p == *pre)//�ҵ�ƥ��Ԫ�أ���Ϊ�����
			break;
	k = p - in;//kΪ��������������е�λ��
	b->lchild = CreateBT1(pre + 1, in, k);//�ݹ���ò�ѯ�����Һ���
	b->rchild = CreateBT1(pre + 1+k, p+1,n-1-k);
	return b;
}
BTNode* CreateBT2(char* post, char* in, int n) {
	BTNode* b;
	char r,*p;
	int k;
	if (n <= 0) return NULL;//�����򲻴���
	b = (BTNode*)malloc(sizeof(BTNode));
	r = *(post + n - 1);
	b->data = r;
	for (p = in;p < in + n;p++)//��������Ѱ����*pԪ����ȵ�Ԫ��
		if (*p == r)//�ҵ�ƥ��Ԫ�أ���Ϊ�����
			break;
	k = p - in;//kΪ��������������е�λ��
	b->lchild = CreateBT2(post, in, k);//�ݹ���ò�ѯ�����Һ���
	b->rchild = CreateBT2(post + k, p + 1, n - 1 - k);
	return b;
}
//���ű�ʾ��Ϊbtree���е�DispBTNode();
void DispBTree1(BTNode*b) {//��͹��
	BTNode* St[MAXWidth], * p;
	int level[MAXWidth][2], top = -1, n, i, width = 4;//��ʼ�� level[][0]�洢����ȣ�level[][1]�洢�������
	char type;        //���Һ��ӱ�� 
	if (b != NULL) {
		top++;St[top] = b;  //����ջ
		level[top][0] = width;//��¼�������
		level[top][1] = 2;  
		while (top > -1) {
			p = St[top];//ջ������ջ
			n = level[top][0];//��ȡ����
			switch (level[top][1]) {//�������Ը���type
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
				level[top][0] = n + width;//������ӣ���ʾ���뵱ǰ���ͬ��
				level[top][1] = 1;
			}
			if (p->lchild != NULL) {
				top++;
				St[top] = p->lchild;
				level[top][0] = n + width;//������ӣ���ʾ���뵱ǰ���ͬ��
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
	printf("���������%s\n", pre);
	printf("���������%s\n", in);
	printf("���������ű�ʾ����");
	DispBTNode(b);cout << endl;
	printf("�����İ�͹��ʾ����\n");
	DispBTree1(b);cout << endl;

	b = CreateBT2(post, in, n);
	printf("���������%s\n", in);
	printf("���������%s\n", post);
	printf("���������ű�ʾ����");
	DispBTNode(b);cout << endl;
	printf("�����İ�͹��ʾ����\n");
	DispBTree1(b);cout << endl;
	DestroyBT(b);
}