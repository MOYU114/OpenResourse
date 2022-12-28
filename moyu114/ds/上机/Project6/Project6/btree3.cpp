#include"btree3.h"
void CreateBTNode(BTNode*& b, char* str)
{      //由str推出二叉链b
    BTNode* St[MaxSize], * p;//建立栈

    int top = -1, k, j = 0;
    char ch;
    b = NULL;		//建立的二叉链初始时为空
    ch = str[j];
    p = (BTNode*)malloc(sizeof(BTNode));
    p->data = ch;  p->lchild = p->rchild = NULL;
    while (ch != '\0')  	//str未扫描完时循环
    {
        switch (ch)
        {
        case '(': top++; St[top] = p; k = 1; break;	//可能有左孩子结点，进栈
        case ')': top--;  break;
        case ',': k = 2;  break; 			//后面为右孩子结点
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
        DestroyBT(b->lchild);//销毁左子树
        DestroyBT(b->rchild);//销毁右子树
        free(b);      //只剩下一个结点*b，直接释放
    }
}
BTNode* FindNode(BTNode* b, ElemType x)
{
    BTNode* p;
    //两个出口
    if (b == NULL) return NULL;//b为空
    else if (b->data == x) return b;//找到了值
    //两个过程
    else
    {
        p = FindNode(b->lchild, x);//寻找b左子树中是否有所需值
        if (p != NULL) return p;//不为空，证明有，返回p
        else return FindNode(b->rchild, x);//否则寻找右子树
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
    int lchilddep, rchilddep;
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
        printf("%c", b->data);
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
