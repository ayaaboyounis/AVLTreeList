
import math
import random

"""A class represnting a node in an AVL tree"""


class AVLNode(object):
    """Constructor, you are allowed to add more fields. 

    @type value: str
    @param value: data of your node
    """

    def __init__(self, value):
        self.value = value
        self.left = Virtual
        self.right = Virtual
        self.parent = None
        self.height = 0
        self.size = 1

    """returns the left child
    @rtype: AVLNode
    @returns: the left child of self, None if there is no left child
    
    Complexity: O(1)
    
    """

    def getLeft(self):
        return self.left

    """returns the right child

    @rtype: AVLNode
    @returns: the right child of self, None if there is no right child
    
    Complexity:O(1)
    """

    def getRight(self):
        return self.right

    """returns the parent 

    @rtype: AVLNode
    @returns: the parent of self, None if there is no parent
    
    Complexity:O(1)
    """

    def getParent(self):
        return self.parent

    """return the value

    @rtype: str
    @returns: the value of self, None if the node is virtual
    
    Complexity:O(1)
    """

    def getValue(self):
        return self.value

    """returns the height

    @rtype: int
    @returns: the height of self, -1 if the node is virtual
    
    Complexity:O(1)
    """

    def getHeight(self):
        if(self.isRealNode()):
            return self.height
        return -1
    
    """returns the size

    @rtype: int
    @returns: the size of self
    
    Complexity:O(1)
    """

    def getSize(self):
        return self.size
    
    """finds the rank of self

    @rtype: int
    @returns: the rank of self
    
    Complexity:O(log n)
    """

    def getRank(self):

        r = self.getLeft().size + 1
        y = self

        while(y.getParent() is not None):
            if(y.getParent().getRight() is y):
                r = r + y.getParent().getLeft().getSize()+1
            y = y.getParent()

        return r

    """sets left child

    @type node: AVLNode
    @param node: a node
    
    Complexity:O(1)
    """

    def setLeft(self, node):
        self.left = node

    """sets right child

    @type node: AVLNode
    @param node: a node
    
    Complexity:O(1)
    """

    def setRight(self, node):
        self.right = node

    """sets parent

    @type node: AVLNode
    @param node: a node
    
    Complexity:O(1)
    """

    def setParent(self, node):
        self.parent = node

    """sets value

    @type value: str
    @param value: data
    
    Complexity:O(1)
    """

    def setValue(self, value):
        self.value = value

    """sets the height of the node

    @type h: int
    @param h: the height
    
    Complexity:O(1)
    """

    def setHeight(self, h):
        self.height = h
        
    """sets the size of the node

    @type s: int
    @param s: the size
    
    Complexity:O(1)
    """     

    def setSize(self, s):
        self.size = s
    
    
    """sets the rank of the node

    @type r: int
    @param r: the rank
    
    Complexity:O(1)
    """     

    def setrRank(self, r):
        self.rank = r

    """returns whether self is not a virtual node 

    @rtype: bool
    @returns: False if self is a virtual node, True otherwise.
    
    Complexity:O(1)
    """

    def isRealNode(self):
        return self is not Virtual
    
    """returns the Balance Factor of the node

    @rtype: int
    @retruns: the Balance Factor of the node,if its not a real node return 0
    
    Complexity:O(1)
    """

    def getBF(self):
        if self.isRealNode():
            return self.getLeft().getHeight() - self.getRight().getHeight()
        else:
            return 0
        
        
    """updates the height of a node

    @rtype: int
    @returns: the height
    
    Complexity:O(1)
    """     

    def UpdateHeight(self):

        self.setHeight(1+max(self.getLeft().getHeight(),
                       self.getRight().getHeight()))
        return self.getHeight()
    
    """counts how many sons the node has (0,1,2)

    @rtype: int
    @returns : the number of sons the node has
    
    Complexity:O(1)
    """

    def CountSons(self):
        Sum = 0
        if(self.getRight() is not None and self.getRight().isRealNode()):
            Sum = Sum+1
        if(self.getLeft() is not None and self.getLeft().isRealNode()):
            Sum = Sum+1

        return Sum
    
    """Checks if the node was a left or right node, and replaces it with v

    @type v: AVLNode
    @param v: a node
    
    Complexity:O(1)
    """

    def CheckChange(self, v):
        Parent = self.getParent()
        if(Parent.getRight() is self):
            Parent.setRight(v)

        if(Parent.getLeft() is self):
            Parent.setLeft(v)

        if(v.isRealNode()):
            v.setParent(Parent)

"""A class represnting a virtual node in an AVL tree"""

class VirtualNode(AVLNode):
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None
        self.parent = None
        self.height = -1
        self.rank = -1
        self.size = 0

    """returns height of virtual node

    @rtype: int
    @returns: -1
    
    Complexity:O(1)
    """
    
    def getHeight(self):
        return -1
    
    """returns the left of virtual node

    @rtype: None
    @returns: None
    
    Complexity:O(1)
    """

    def getLeft(self):
        return None
    
    """returns the right of virtual node

    @rtype: None
    @returns: None
    
    Complexity:O(1)
    """

    def getRight(self):
        return None
    
    """returns the BF of virtual node

    @rtype: int
    @returns: 0
    
    Complexity:O(1)
    """

    def getBF(self):
        return 0


Virtual = VirtualNode()

"""
A class implementing the ADT list, using an AVL tree.
"""


class AVLTreeList(object):

    """
    Constructor, you are allowed to add more fields.  

    """

    def __init__(self):
        self.root = Virtual
        self.len = 0
        self.first=Virtual
        self.last=Virtual
    """returns 
    whether the list is empty

    @rtype: bool
    @returns: True if the list is empty, False otherwise
    
    Complexity: O(1)
    """

    def empty(self):
        if self.getRoot().getSize() > 0:
            return False
        return True

    """retrieves the value of the i'th item in the list

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: index in the list
    @rtype: str
    @returns: the the value of the i'th item in the list
    
    Complexity: O(log n)
    """

    def retrieve(self, i):
        return self.Select(i+1).getValue()
    
    """returns the node with rank k

    @type k: int
    @pre: 1 <= k <= self.length()
    @param k: rank in the list
    @rtype: a node
    @returns: the node of rank k
    
    Complexity: O(log n)
    """

    def Select(self, k):
        v = self.root

        r = v.getLeft().getSize()+1
        if k == r:
            return v
        else:
            if k < r:
                LTree = AVLTreeList()
                LTree.UpdateRoot(v.getLeft())
                LTree.len = v.getLeft().getSize()
                return LTree.Select(k)
            else:
                RTree = AVLTreeList()
                RTree.UpdateRoot(v.getRight())
                RTree.len = v.getRight().getSize()
                return RTree.Select(k-r)

    """inserts val at position i in the list

    @type i: int
    @pre: 0 <= i <= self.length()
    @param i: The intended index in the list to which we insert val
    @type val: str
    @param val: the value we inserts
    @rtype: list
    @returns: the number of rebalancing operation due to AVL rebalancing
    
    Complexity:O(log n)
    """

    def insert(self, i, val):

        newNode = AVLNode(val)
        newNode.setrRank(i)
        n = self.length()

        
        if(i==0):
            self.first=newNode
        
        if(i==n):
            self.last=newNode
            
        #if the tree is empty
        if(n == 0):
            self.root = newNode
            self.len = 1
            return 0
        
        #if we insert at the end of the list
        
        if(i == n):
            v = self.root
            while(v.getRight() is not None and v.getRight().isRealNode()):
                v = v.getRight()
            v.setRight(newNode)
            newNode.setParent(v)

        else:
            
            v = self.Select(i+1)
            
            #if the current node at index i doesnt have left son -> set the new node as its left son
            
            if (not v.getLeft().isRealNode()):
                v.setLeft(newNode)
                newNode.setParent(v)
                
            #if the current node at index i has left son -> set the new node as right son of the predecessor
            
            else:
                Predecessor = self.Select(i)
                Predecessor.setRight(newNode)
                newNode.setParent(Predecessor)

        x = self.rebalance(newNode)
        self.UpdateRoot2()
        self.UpdateSizes(newNode)
        self.len = self.len+1

        return x

    """deletes the i'th item in the list
 
    @type i: int
    @pre: 0 <= i < self.length()
    @param i: The intended index in the list to be deleted
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    
    Complexity: O(log n)
    """

    def delete(self, i):
        n = self.length()
        v = self.Select(i+1)
        SonsNumber = v.CountSons()
        parent = v.getParent()

        #if there is only one node in the list
        
        if(n == 1):
            v = self.root
            self.root = Virtual
            self.first=Virtual
            self.last=Virtual
            self.len = self.len-1
            return 0
        
        
        if(i==n):
            self.last=self.Select(n-1)
        
        if(i==0):
            self.first=self.Select(i+2)


        #if the node we want to delete is the root and it has one son
        
        if(self.root is v and SonsNumber == 1):
            if(SonsNumber == 1):
                vLeft = v.getLeft()
                vRight = v.getRight()

                if(vLeft.isRealNode()):
                    self.root = vLeft

                if(vRight.isRealNode()):
                    self.root = vRight
            self.len = self.len-1
            return 0

        else:
            
            #if the node we want to delete has no sons -> replace with virtual
            
            if(SonsNumber == 0):
                v.CheckChange(Virtual)

            #if the nodewe want to delete has one son -> replace with the son
            
            if(SonsNumber == 1):
                if(v.getRight() is not None and v.getRight().isRealNode()):
                    vRight = v.getRight()
                    v.CheckChange(vRight)

                if(v.getLeft() is not None and v.getLeft().isRealNode()):
                    vLeft = v.getLeft()
                    v.CheckChange(vLeft)

            #if the nodewe want to delete has two sons -> replace with the successor
            if(SonsNumber == 2):
                self.successSwitch(i)

            x = self.DeleteRebalance(parent)
            self.UpdateRoot2()
            self.UpdateSizes(parent)
            self.UpHeights(parent)
            self.len = self.len-1

            return x

    """returns the value of the first item in the list

    @rtype: str
    @returns: the value of the first item, None if the list is empty
    
    Complexity:O(1)
    """

    def first(self):
        v=self.root
        while(v.isRealNode()):
            v=v.getLeft()
        self.first=v
        return v.getValue()

    """returns the value of the last item in the list

    @rtype: str
    @returns: the value of the last item, None if the list is empty
    
    Complexity:O(1)
    """

    def last(self):
        v=self.root
        while(v.isRealNode()):
            v=v.getRight()
        self.last=v
        return v.getValue()

    """returns an array representing list 

    @rtype: list
    @returns: a list of strings representing the data structure
    
    Complexity: O(n)
    """

    def listToArray(self):
        arr = []
        v = self.root
        i = 0
        
        #fill the list recursively 
        
        self.listToArrayRec(i, arr, v)
        return arr
    
    def listToArrayRec(self, i, arr, v):
        if (v is not None and v.isRealNode()):
            i = self.listToArrayRec(i, arr, v.getLeft())
            i = i+1
            arr.insert(i, v.getValue())
            i = self.listToArrayRec(i, arr, v.getRight())

        return i

    """returns the size of the list 

    @rtype: int
    @returns: the size of the list
    
    Complexity: O(1)
    """

    def length(self):
        n = self.len
        return n

    """splits the list at the i'th index

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: The intended index in the list according to whom we split
    @rtype: list
    @returns: a list [left, val, right], where left is an AVLTreeList representing the list until index i-1,
    right is an AVLTreeList representing the list from index i+1, and val is the value at the i'th index.
    
    Complexity: O(log n)
    
    """

    def split(self, i):
        x = self.Select(i+1)
        v = x
        Smaller = AVLTreeList()
        Bigger = AVLTreeList()
        
        #build the left tree
        
        if(x.getLeft().isRealNode()):
            x.getLeft().setParent(None)
            Smaller.UpdateRoot(x.getLeft())
            Smaller.len = x.getLeft().getSize()
            
        #build the right tree
        
        if(x.getRight().isRealNode()):
            x.getRight().setParent(None)
            Bigger.UpdateRoot(x.getRight())
            Bigger.len = x.getRight().getSize()

        f = x.getParent()
        while(f is not None):
            
            #add to the left Tree
            
            if(f.getRight() is x):

                Smaller.insert(0, f.getValue())
                C = AVLTreeList()
                C.UpdateRoot(f.getLeft())
                C.len = f.getLeft().getSize()
                f.getLeft().setParent(None)
                C.concat(Smaller)
                Smaller = C
                
            #add to the right tree 
            
            if(f.getLeft() is x):

                Bigger.insert(Bigger.length(), f.getValue())
                G = AVLTreeList()
                G.UpdateRoot(f.getRight())
                G.len = f.getRight().getSize()
                f.getRight().setParent(None)
                Bigger.concat(G)


            f = f.getParent()
            x = x.getParent()

        result = [Smaller, v, Bigger]

        return result

    """concatenates lst to self

    @type lst: AVLTreeList
    @param lst: a list to be concatenated after self
    @rtype: int
    @returns: the absolute value of the difference between the height of the AVL trees joined
    
    Complexity: O(log n)
    """

    def concat(self, lst):
        
        #if lst is empty -> do nothing
        
        if(lst.getRoot().getSize() == 0):
            return self.getRoot().getHeight()
        
        #if self is empty -> change the root to lst's root
        
        if(self.getRoot().getSize() == 0):
            self.UpdateRoot(lst.getRoot())
            self.len = lst.length()
            return self.getRoot().getHeight()

        v = self.Select(self.length())
        self.delete(self.length()-1)

        Root1 = self.getRoot()
        Root2 = lst.getRoot()

        Height1 = Root1.getHeight()
        Height2 = Root2.getHeight()

        result = abs(Height1-Height2)

        if(Root1.getSize() == 0):

            lst.insert(0, v.getValue())
            self.UpdateRoot(lst.getRoot())
            return result

        else:

            if(Height1 == Height2):
                v.setLeft(Root1)
                v.setRight(Root2)
                Root1.setParent(v)
                Root2.setParent(v)
                v.setParent(None)
                self.UpdateRoot(v)

            else:
                if(Height1 > Height2):
                    while(Root1.getHeight() > Root2.getHeight()):
                        Root1 = Root1.getRight()

                    if(not Root1.isRealNode()):
                        v.setRight(Root2)
                        v.setLeft(Virtual)
                        Root1Parent = self.Select(self.length())
                        v.setParent(Root1Parent)
                        Root1Parent.setRight(v)
                    else:
                        v.setLeft(Root1)
                        v.setRight(Root2)
                        Root1Parent = Root1.getParent()
                        v.setParent(Root1Parent)
                        Root1Parent.setRight(v)

                    Root1.setParent(v)
                    Root2.setParent(v)

                else:
                    while(Root2.getHeight() > Root1.getHeight()):
                        Root2 = Root2.getLeft()

                    if(not Root2.isRealNode()):
                        v.setLeft(Root1)
                        v.setRight(Virtual)
                        Root2Parent = lst.Select(1)
                        v.setParent(Root2Parent)
                        Root2Parent.setLeft(v)
                    else:
                        v.setLeft(Root1)
                        v.setRight(Root2)
                        Root2Parent = Root2.getParent()
                        v.setParent(Root2Parent)
                        Root2Parent.setLeft(v)

                    Root1.setParent(v)
                    Root2.setParent(v)

        self.JoinRebalance(v)
        self.UpdateRoot2()
        self.UpdateSizes(v)
        self.len = self.length() + lst.length()

        return result

    """searches for a *value* in the list

    @type val: str
    @param val: a value to be searched
    @rtype: int
    @returns: the first index that contains val, -1 if not found.
    
    Complexity: O(n)
    """

    def search(self, val):

        v = self.root
        i = -1
        #search for the value recursively
        return self.RecSearch(v, val, i)

    def RecSearch(self, v, val, i):

        if(v is not None and v.isRealNode()):

            i = self.RecSearch(v.getLeft(), val, i)

            if(v.getValue() == val):
                return v.getRank()-1

            i = self.RecSearch(v.getRight(), val, i)

        return i

    """returns the root of the tree representing the list

    @rtype: AVLNode
    @returns: the root, None if the list is empty
    
    Complexity: O(1)
    """

    def getRoot(self):
        return self.root
    
    """returns an array representing list 

    @type newRoot: AVLNode
    @param newRoot: the new root
    
    Complexity: O(1)
    """

    def UpdateRoot(self, NewRoot):
        self.root = NewRoot
        
    """rebalances the Tree after an insert

    @rtype: int
    @param v: the node we just inserted
    @returns: the number of rebalancing operations
    
    Complexity: O(log n)
    """        

    def rebalance(self, v):
        Sum = 0
        y = v.getParent()

        while(y is not None):
            bf = y.getBF()

            height = y.getHeight()
            Newheight = 1+max(y.getLeft().getHeight(),
                              y.getRight().getHeight())

            if(abs(bf) < 2):
                
                #if the height didnt change -> terminate
                
                if(height == Newheight):
                    return Sum

                else:
                    y.UpdateHeight()
                    y = y.getParent()
                    Sum = Sum+1

            else:
                if(bf == -2):
                    bfR = y.getRight().getBF()

                    #case (-2,-1) -> left rotation
                    
                    if(bfR == -1):
                        self.lRotation(y)
                        return Sum+1
                    
                    #case (-2,1) -> right left rotation
                    
                    else:

                        self.rRotation(y.getRight())
                        self.lRotation(y)
                        return Sum+2

                else:
                    bfL = y.getLeft().getBF()

                    #case (2,1) -> right rotation
                    
                    if(bfL == 1):
                        self.rRotation(y)
                        return Sum+1
                    
                    #case (2,-1) -> left right rotation
                    
                    else:

                        self.lRotation(y.getLeft())
                        self.rRotation(y)
                        return Sum+2
        return Sum
    
    """Rotates left

    @rtype: a node
    @param v: a node
    @returns: the node we rotated on
    
    Complexity: O(1)
    """     

    def lRotation(self, x):
        y = x.getRight()
        T2 = y.getLeft()
        xParent = x.getParent()

        y.setLeft(x)
        x.setParent(y)

        x.setRight(T2)
        if(T2 is not None and T2.isRealNode()):
            T2.setParent(x)

        y.setParent(xParent)

        if(xParent is not None):
            if(xParent.getLeft() is x):
                xParent.setLeft(y)
            else:
                xParent.setRight(y)
        
        x.setHeight(max(x.getLeft().getHeight(), x.getRight().getHeight()) + 1)
        y.setHeight(max(y.getLeft().getHeight(), y.getRight().getHeight()) + 1)

        x.setSize(1+x.getLeft().getSize()+x.getRight().getSize())
        y.setSize(1+y.getLeft().getSize()+y.getRight().getSize())
        return y
    
    """Rotates right

    @rtype: a node
    @param v: a node
    @returns: the node we rotated on
    
    Complexity: O(1)
    """     

    def rRotation(self, y):
        x = y.getLeft()
        T2 = x.getRight()
        yParent = y.getParent()

        x.setRight(y)
        y.setParent(x)

        y.setLeft(T2)
        if(T2 is not None and T2.isRealNode()):
            T2.setParent(y)

        x.setParent(yParent)

        if(yParent is not None):
            if(yParent.getLeft() is y):
                yParent.setLeft(x)
            else:
                yParent.setRight(x)

        y.setHeight(max(y.getLeft().getHeight(), y.getRight().getHeight()) + 1)
        x.setHeight(max(x.getLeft().getHeight(), x.getRight().getHeight()) + 1)
        y.setSize(1+y.getLeft().getSize()+y.getRight().getSize())
        x.setSize(1+x.getLeft().getSize()+x.getRight().getSize())

        return x
    
    """Updates sizes from a given node to the root

    @type v: AVLNode
    @param v: a node
    
    Complexity: O(log n)
    """     

    def UpdateSizes(self, v):
        Node = v
        while (Node is not None and Node.isRealNode()):
            Node.setSize(1+Node.getLeft().getSize()+Node.getRight().getSize())
            Node = Node.getParent()
            
     
    """Updates the root

    Complexity: O(log n)
    """                 

    def UpdateRoot2(self):
        curRoot = self.root
        while(curRoot.getParent() is not None):
            curRoot = curRoot.getParent()
        self.UpdateRoot(curRoot)
        
        
    """rebalances the Tree after a delete

    @rtype: int
    @param v: the node we want to delete
    @returns: the number of rebalancing operations
    
    Complexity: O(log n)
    """                

    def DeleteRebalance(self, v):
        Sum = 0

        while(v is not None):
            bf = v.getBF()

            height = v.getHeight()
            Newheight = 1+max(v.getLeft().getHeight(),
                              v.getRight().getHeight())

            if (abs(bf) < 2):
                
                #if height hasnt changed -> terminate
                
                if (height == Newheight):
                    return Sum
                else:
                    v.UpdateHeight()
                    v = v.getParent()
                    Sum = Sum+1

            else:

                if(v.getBF() == -2):
                    
                    #case (-2,-1) or (-2,0) -> left rotation
                    
                    if(v.getRight().getBF() == -1 or v.getRight().getBF() == 0):
                        self.lRotation(v)
                        self.UpdateSizes(v)
                        Sum = Sum+1
                        
                    #case (-2,1) -> right left rotation 
                    
                    else:
                        self.rRotation(v.getRight())
                        self.lRotation(v)

                        Sum = Sum+2

                if(v.getBF() == 2):
                    
                    #case (2,1) or (2,0) -> right rotation
                    
                    if(v.getLeft().getBF() == 1 or v.getLeft().getBF() == 0):
                        self.rRotation(v)
                        Sum = Sum+1
                        
                    #case (2,-1) -> left right rotation
                    
                    else:
                        self.lRotation(v.getLeft())
                        self.rRotation(v)
                        Sum = Sum+2
                v = v.getParent()
        return Sum
    
    """switches between node in index i and its successor

    @type i: int
    @param i: the node of the node
    
    Complexity: O(log n)
    """        

    def successSwitch(self, i):
        y = self.Select(i+2)
        x = self.Select(i+1)

        xLeft = x.getLeft()
        xRight = x.getRight()

        xParent = x.getParent()
        if(y is xRight):
            if(self.getRoot() is x):
                self.UpdateRoot(y)
                y.setLeft(xLeft)
                xLeft.setParent(y)
                self.UpdateSizes(y)
                self.UpHeights(y)
                y.setParent(None)
                return
            x.CheckChange(y)
            y.setLeft(xLeft)
            xLeft.setParent(y)
            self.UpdateSizes(y)
            self.UpHeights(y)
            return

        yParent = y.getParent()
        yRight = y.getRight()

        yParent.setLeft(yRight)
        yRight.setParent(yParent)

        y.setLeft(xLeft)
        y.setRight(xRight)
        y.setParent(xParent)

        if(xParent is None):
            self.root = y
        else:
            x.CheckChange(y)

        xLeft.setParent(y)
        xRight.setParent(y)

        if(yRight.isRealNode()):
            self.UpdateSizes(yRight)
            self.UpHeights(yRight)

        else:
            self.UpdateSizes(yParent)
            self.UpHeights(yParent)

    """rebalances the Tree after joining 2 trees

    @type v: AVLNode
    @param v: the node we are going to start rebalancing from
    
    Complexity: O(log n)
    """    
    def JoinRebalance(self, v):
        y = v.getParent()

        while(y is not None):
            bf = y.getBF()

            height = y.getHeight()
            Newheight = 1+max(y.getLeft().getHeight(),
                              y.getRight().getHeight())

            if(abs(bf) < 2):

                if(height == Newheight):
                    return

                else:
                    y.UpdateHeight()
                    y = y.getParent()

            else:

                if(bf == -2):
                    bfR = y.getRight().getBF()

                    if(bfR == -1):
                        self.lRotation(y)
                        return
                    else:

                        if(bfR == 1):
                            self.rRotation(y.getRight())
                            self.lRotation(y)
                            return
                        else:
                            self.lRotation(y)
                            y = y.getParent()

                else:
                    bfL = y.getLeft().getBF()

                    if(bfL == 1):
                        self.rRotation(y)
                        return
                    else:
                        if(bfL == -1):
                            self.lRotation(y.getLeft())
                            self.rRotation(y)
                            return
                        else:
                            self.rRotation(y)
                            y = y.getParent()


    """Updates heights starting from v

    @type v: AVLNode
    @param v: the node we want to start updating from
    
    Complexity: O(log n)
    """    
    def UpHeights(self, v):
        node = v
        while(node is not None):
            node.setHeight(1+max(v.getLeft().getHeight(),
                           v.getRight().getHeight()))
            node = node.getParent()

