#include<iostream>
using namespace std;
typedef char ElemType;
#define MaxSize 50
typedef struct node
{
    ElemType data;
    struct node* lchild,* rchild;
}   BTNode;
void CreateBTNode(BTNode*& b,char* str)
{      //��str�Ƴ�������b
    BTNode* St[MaxSize],* p;//����ջ
    int top = -1,k, j = 0;
    char ch;
    b = NULL;		//�����Ķ�������ʼʱΪ��
    ch = str[j];
    while (ch != '\0')  	//strδɨ����ʱѭ��
    {
        switch (ch)
        {
        case '(': top++; St[top] = p; k = 1; break;	//���������ӽ�㣬��ջ
        case ')': top--;  break;
        case '��': k = 2;  break; 			//����Ϊ�Һ��ӽ��
        default:        		 //�������ֵ
            p = (BTNode*)malloc(sizeof(BTNode));
            p->data = ch;  p->lchild = p->rchild = NULL;
            if (b == NULL)    	//pΪ�������ĸ����
                b = p;
            else    			//�ѽ��������������
            {
                switch (k)
                {
                case 1:  St[top]->lchild = p;  break;
                case 2:  St[top]->rchild = p;  break;
                }
            }
        }
        j++;  ch = str[j];		//����ɨ��str
    }
}
void DestroyBT(BTNode*& b)
{
    if (b == NULL) return;
    else
    {
        DestroyBT(b->lchild);
        DestroyBT(b->rchild);
        free(b);      //ʣ��һ�����*b��ֱ���ͷ�
    }
}
BTNode* FindNode(BTNode* b,ElemType x)
{
    BTNode* p;
    if (b == NULL) return NULL;
    else if (b->data == x) return b;
    else
    {
        p = FindNode(b->lchild,x);
        if (p != NULL) return p;
        else return FindNode(b->rchild,x);
    }
}
BTNode* LchildNode(BTNode* p)
{
    return p->lchild;
}

BTNode* RchildNode(BTNode* p)
{
    return p->rchild;
}
int BTNodeDepth(BTNode* b)
{
    int lchilddep,rchilddep;
    if (b == NULL) return(0); 	//�����ĸ߶�Ϊ0
    else
    {
        lchilddep = BTNodeDepth(b->lchild);
        //���������ĸ߶�Ϊlchilddep
        rchilddep = BTNodeDepth(b->rchild);
        //���������ĸ߶�Ϊrchilddep
        return(lchilddep > rchilddep) ? (lchilddep + 1) : (rchilddep + 1);
    }
}
void DispBTNode(BTNode* b)
{
    if (b != NULL)
    {
        printf("%c",b->data);
        if (b->lchild != NULL || b->rchild != NULL)
        {
            printf("(");
            DispBTNode(b->lchild);//�ݹ鴦��������
            if (b->rchild != NULL) printf("��");
            DispBTNode(b->rchild);//�ݹ鴦��������
            printf(")");
        }
    }
}

void PreOrder(BTNode* b)//�������
{
    if (b != NULL)
    {
        printf("%c ",b->data); 	//���ʸ����
        PreOrder(b->lchild);
        PreOrder(b->rchild);
    }
}

void InOrder(BTNode* b)//�������
{
    if (b != NULL)
    {
        InOrder(b->lchild);
        printf("%c ",b->data); 	//���ʸ����
        InOrder(b->rchild);
    }
}

void PostOrder(BTNode* b)//�������
{
    if (b != NULL)
    {
        PostOrder(b->lchild);
        PostOrder(b->rchild);
        printf("%c ",b->data); 	//���ʸ����
    }
}
void PreOrder1(BTNode* b)
{
    BTNode* p;
    SqStack* st;				//����ջָ��st
    InitStack(st);			//��ʼ��ջst
    if (b != NULL)
    {
        Push(st��b);			//������ջ
        while (!StackEmpty(st))		//ջ��Ϊ��ʱѭ��
        {
            Pop(st��p);			//��ջ���p��������
            printf("%c ",p->data);
            if (p->rchild != NULL)	//���Һ���ʱ�����ջ
                Push(st��p->rchild);
            if (p->lchild != NULL)	//������ʱ�����ջ
                Push(st��p->lchild);
        }
        printf("\n");
    }
    DestroyStack(st);			//����ջ
}
typedef struct
{
    BTNode* data[MaxSize];	//��Ŷ���Ԫ��
    int front, rear;		//��ͷ�Ͷ�βָ��
} SqQueue;			//���ζ�������
void LevelOrder(BTNode* b)
{
    BTNode* p;
    SqQueue* qu;			//���廷�ζ���ָ��
    InitQueue(qu);			//��ʼ������
    enQueue(qu,b);			//�����ָ��������
    while (!QueueEmpty(qu))		//�Ӳ�Ϊ��ѭ��
    {
        deQueue(qu��p);		//���ӽ��p
        printf("%c ",p->data);		//���ʽ��p
        if (p->lchild != NULL)		//������ʱ�������
            enQueue(qu��p->lchild);
        if (p->rchild != NULL)		//���Һ���ʱ�������
            enQueue(qu��p->rchild);
    }
}
