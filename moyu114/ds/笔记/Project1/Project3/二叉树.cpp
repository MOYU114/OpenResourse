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
{      //由str推出二叉链b
    BTNode* St[MaxSize],* p;//建立栈
    int top = -1,k, j = 0;
    char ch;
    b = NULL;		//建立的二叉链初始时为空
    ch = str[j];
    while (ch != '\0')  	//str未扫描完时循环
    {
        switch (ch)
        {
        case '(': top++; St[top] = p; k = 1; break;	//可能有左孩子结点，进栈
        case ')': top--;  break;
        case '，': k = 2;  break; 			//后面为右孩子结点
        default:        		 //遇到结点值
            p = (BTNode*)malloc(sizeof(BTNode));
            p->data = ch;  p->lchild = p->rchild = NULL;
            if (b == NULL)    	//p为二叉树的根结点
                b = p;
            else    			//已建立二叉树根结点
            {
                switch (k)
                {
                case 1:  St[top]->lchild = p;  break;
                case 2:  St[top]->rchild = p;  break;
                }
            }
        }
        j++;  ch = str[j];		//继续扫描str
    }
}
void DestroyBT(BTNode*& b)
{
    if (b == NULL) return;
    else
    {
        DestroyBT(b->lchild);
        DestroyBT(b->rchild);
        free(b);      //剩下一个结点*b，直接释放
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
    if (b == NULL) return(0); 	//空树的高度为0
    else
    {
        lchilddep = BTNodeDepth(b->lchild);
        //求左子树的高度为lchilddep
        rchilddep = BTNodeDepth(b->rchild);
        //求右子树的高度为rchilddep
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
            DispBTNode(b->lchild);//递归处理左子树
            if (b->rchild != NULL) printf("，");
            DispBTNode(b->rchild);//递归处理右子树
            printf(")");
        }
    }
}

void PreOrder(BTNode* b)//先序遍历
{
    if (b != NULL)
    {
        printf("%c ",b->data); 	//访问根结点
        PreOrder(b->lchild);
        PreOrder(b->rchild);
    }
}

void InOrder(BTNode* b)//中序遍历
{
    if (b != NULL)
    {
        InOrder(b->lchild);
        printf("%c ",b->data); 	//访问根结点
        InOrder(b->rchild);
    }
}

void PostOrder(BTNode* b)//后序遍历
{
    if (b != NULL)
    {
        PostOrder(b->lchild);
        PostOrder(b->rchild);
        printf("%c ",b->data); 	//访问根结点
    }
}
void PreOrder1(BTNode* b)
{
    BTNode* p;
    SqStack* st;				//定义栈指针st
    InitStack(st);			//初始化栈st
    if (b != NULL)
    {
        Push(st，b);			//根结点进栈
        while (!StackEmpty(st))		//栈不为空时循环
        {
            Pop(st，p);			//退栈结点p并访问它
            printf("%c ",p->data);
            if (p->rchild != NULL)	//有右孩子时将其进栈
                Push(st，p->rchild);
            if (p->lchild != NULL)	//有左孩子时将其进栈
                Push(st，p->lchild);
        }
        printf("\n");
    }
    DestroyStack(st);			//销毁栈
}
typedef struct
{
    BTNode* data[MaxSize];	//存放队中元素
    int front, rear;		//队头和队尾指针
} SqQueue;			//环形队列类型
void LevelOrder(BTNode* b)
{
    BTNode* p;
    SqQueue* qu;			//定义环形队列指针
    InitQueue(qu);			//初始化队列
    enQueue(qu,b);			//根结点指针进入队列
    while (!QueueEmpty(qu))		//队不为空循环
    {
        deQueue(qu，p);		//出队结点p
        printf("%c ",p->data);		//访问结点p
        if (p->lchild != NULL)		//有左孩子时将其进队
            enQueue(qu，p->lchild);
        if (p->rchild != NULL)		//有右孩子时将其进队
            enQueue(qu，p->rchild);
    }
}
