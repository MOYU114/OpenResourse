#include"btree.h"

int main() {
    char str[] = { "A(B(D,E(H(J,K(L,M(,N))))),C(F,G(,I)))" };
    BTNode* b, * p, * lp, * rp;
    cout << "（1）创建二叉树" << endl;
    CreateBTNode(b, str);
    cout << "（2）输出二叉树b" << endl;
    DispBTNode(b);
    cout << endl;
    cout << "（3）输出'H'结点的左、右孩子值" << endl;
    p = FindNode(b, 'H');
    if (p != NULL) {
        lp = LchildNode(p);
        rp = RchildNode(p);
        if (lp != NULL)
            cout << "H的左孩子为：" << lp->data<<endl;
        else
            cout<< "H无左孩子" << endl;
        if (rp != NULL)
            cout << "H的右孩子为：" << rp->data << endl;
        else
            cout << "H无右孩子" << endl;
    }
    else
        cout << "H无左右孩子" << endl;
    cout << "（4）输出二叉树b的高度" << endl;
    cout << "b的高度为：" << BTNodeDepth(b) << endl;
    cout << "（5）释放二叉树b" << endl;
    DestroyBT(b);
}