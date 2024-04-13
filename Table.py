from Biblio import *

def createSubMenu(parent, label, commands):
    menu = Menu(parent, tearoff = 0)
    parent.add_cascade(label=label,menu=menu)
    for action in commands:
        menu.add_command(label=action, command=commands[action])
    applyStyle(menu)
    return menu

class ColumnHeader(Canvas):
    """Class that takes it's size and rendering from a parent table
        and column names from the table model."""

    def __init__(self, parent=None, table=None, bg='gray25'):
        Canvas.__init__(self, parent, bg=bg, width=500, height=25)
        self.thefont = 'Arial 14'
        self.textcolor = 'white'
        self.bgcolor = bg
        if table != None:
            self.table = table
            self.model = self.table.model
            if util.check_multiindex(self.model.df.columns) == 1:
                self.height = 40
            else:
                self.height = self.table.rowheight
            self.config(width=self.table.width, height=self.height)
            self.columnlabels = self.model.df.columns
            self.draggedcol = None
            self.bind('<Button-1>',self.handle_left_click)
            self.bind("<ButtonRelease-1>", self.handle_left_release)
            self.bind('<B1-Motion>', self.handle_mouse_drag)
            self.bind('<Motion>', self.handle_mouse_move)
            self.bind('<Shift-Button-1>', self.handle_left_shift_click)
            self.bind('<Control-Button-1>', self.handle_left_ctrl_click)
            self.bind("<Double-Button-1>",self.handle_double_click)
            self.bind('<Leave>', self.leave)
            if self.table.ostyp=='darwin':
                #For mac we bind Shift, left-click to right click
                self.bind("<Button-2>", self.handle_right_click)
                self.bind('<Shift-Button-1>',self.handle_right_click)
            else:
                self.bind("<Button-3>", self.handle_right_click)
            self.thefont = self.table.thefont
            self.wrap = False
            self.setDefaults()
        return

    def setDefaults(self):
        self.colselectedcolor = '#0099CC'
        self.sort_ascending = 1
        return

    def redraw(self, align='w'):
        """Redraw column header"""

        df = self.model.df
        multiindex = util.check_multiindex(df.columns)
        wrap = self.wrap
        cols = self.model.getColumnCount()
        colwidths = self.table.columnwidths
        scale = self.table.getScale() * 1.5
        self.height = self.table.rowheight

        if wrap is True:
            #set height from longest column wrapped
            try:
                c = list(df.columns.map(str).str.len())
            except:
                c = [len(str(i)) for i in df.columns]
            idx = c.index(max(c))
            longest = str(df.columns[idx].encode('utf-8').decode('utf-8'))
            if longest in colwidths:
                cw = colwidths[longest]
            else:
                cw = self.table.cellwidth

            tw,l = util.getTextLength(longest, cw)
            tr = len(textwrap.wrap(longest, l))
            if tr > 1:
                self.height = tr*self.height
            #print (tr, longest, textwrap.wrap(longest, l))

        if self.height>250:
            self.height=250

        self.tablewidth = self.table.tablewidth
        self.thefont = self.table.thefont
        self.configure(scrollregion=(0,0,
                                     self.table.tablewidth+self.table.x_start,
                                     self.height))
        self.config(height=self.height, bg=self.bgcolor)
        self.delete('gridline','text')
        self.delete('rect')
        self.delete('dragrect')
        self.atdivider = None
        font = self.thefont
        anchor = align
        pad = 5

        x_start = self.table.x_start
        if cols == 0:
            return

        if multiindex == 1:
            anchor = 'nw'
            y=2
            levels = df.columns.levels
            h = self.height
            self.height *= len(levels)
            y=3
        else:
            levels = [df.columns.values]
            h = self.height
            y = h/2
        i=0
        #iterate over index levels
        col=0
        for level in levels:
            values = df.columns.get_level_values(i)
            for col in self.table.visiblecols:
                colname = values[col]
                try:
                    colstr = colname.encode('utf-8','ignore').decode('utf-8')
                except:
                    colstr = str(colname)

                if colstr in colwidths:
                    w = colwidths[colstr]
                else:
                    w = self.table.cellwidth

                if w<=8:
                    colname=''
                x = self.table.col_positions[col]
                if anchor in ['w','nw']:
                    xt = x+pad
                elif anchor == 'e':
                    xt = x+w-pad
                elif anchor == 'center':
                    xt = x+w/2

                colname = colstr
                tw,length = util.getTextLength(colstr, w-pad, font=font)
                #print (colstr, tw)
                if wrap is True:
                    colname = textwrap.fill(colstr, length-1)
                    y=3
                    anchor = 'nw'
                else:
                    colname = colname[0:int(length)]

                line = self.create_line(x, 0, x, self.height, tag=('gridline', 'vertline'),
                                     fill='white', width=1)
                self.create_text(xt,y,
                                    text=colname,
                                    fill=self.textcolor,
                                    font=self.thefont,
                                    tag='text', anchor=anchor)

            x = self.table.col_positions[col+1]
            self.create_line(x,0, x, self.height, tag='gridline',
                            fill='white', width=2)
            i+=1
            y=y+h-2
        self.config(height=self.height)
        return

    def handle_left_click(self,event):
        """Does cell selection when left mouse button is clicked"""

        self.delete('rect')
        self.table.delete('entry')
        self.table.delete('multicellrect')
        colclicked = self.table.get_col_clicked(event)
        if colclicked == None:
            return
        #set all rows for plotting if no multi selection
        if len(self.table.multiplerowlist) <= 1:
            self.table.allrows = True

        self.table.setSelectedCol(colclicked)
        if self.atdivider == 1:
            return
        self.drawRect(self.table.currentcol)
        #also draw a copy of the rect to be dragged
        self.draggedcol = None
        self.drawRect(self.table.currentcol, tag='dragrect',
                        color='lightblue', outline='white')
        if hasattr(self, 'rightmenu') and self.rightmenu != None:
            self.rightmenu.destroy()
        #finally, draw the selected col on the table
        self.table.drawSelectedCol()
        self.table.drawMultipleCells()
        self.table.drawMultipleRows(self.table.multiplerowlist)
        return

    def handle_left_release(self,event):
        """When mouse released implement resize or col move"""

        self.delete('dragrect')
        #if ctrl selection return
        if len(self.table.multiplecollist) > 1:
            return
        #column resized
        if self.atdivider == 1:
            x = int(self.canvasx(event.x))
            col = self.nearestcol
            x1,y1,x2,y2 = self.table.getCellCoords(0,col)
            newwidth = x - x1
            if newwidth < 5:
                newwidth=5
            self.table.resizeColumn(col, newwidth)
            self.table.delete('resizeline')
            self.delete('resizeline')
            self.delete('resizesymbol')
            self.atdivider = 0
            return
        self.delete('resizesymbol')
        #move column
        if self.draggedcol != None and self.table.currentcol != self.draggedcol:
            self.model.moveColumn(self.table.currentcol, self.draggedcol)
            self.table.setSelectedCol(self.draggedcol)
            self.table.redraw()
            self.table.drawSelectedCol(self.table.currentcol)
            self.drawRect(self.table.currentcol)
        return

    def handle_right_click(self, event):
        """respond to a right click"""

        if self.table.enable_menus == False:
            return
        colclicked = self.table.get_col_clicked(event)
        multicollist = self.table.multiplecollist
        if len(multicollist) > 1:
            pass
        else:
            self.handle_left_click(event)
        self.rightmenu = self.popupMenu(event)
        return

    def handle_mouse_drag(self, event):
        """Handle column drag, will be either to move cols or resize"""

        x=int(self.canvasx(event.x))
        if self.atdivider == 1:
            self.table.delete('resizeline')
            self.delete('resizeline')
            self.table.create_line(x, 0, x, self.table.rowheight*self.table.rows,
                                width=2, fill='gray', tag='resizeline')
            self.create_line(x, 0, x, self.height,
                                width=2, fill='gray', tag='resizeline')
            return
        else:
            w = self.table.cellwidth
            self.draggedcol = self.table.get_col_clicked(event)
            coords = self.coords('dragrect')
            if len(coords)==0:
                return
            x1, y1, x2, y2 = coords
            x=int(self.canvasx(event.x))
            y = self.canvasy(event.y)
            self.move('dragrect', x-x1-w/2, 0)

        return

    def within(self, val, l, d):
        """Utility funtion to see if val is within d of any
            items in the list l"""

        for v in l:
            if abs(val-v) <= d:
                return v
        return None

    def leave(self, event):
        """Mouse left canvas event"""
        self.delete('resizesymbol')
        return

    def handle_mouse_move(self, event):
        """Handle mouse moved in header, if near divider draw resize symbol"""

        if len(self.model.df.columns) == 0:
            return
        self.delete('resizesymbol')
        w = self.table.cellwidth
        h = self.height
        x_start = self.table.x_start
        #x = event.x
        x = int(self.canvasx(event.x))
        if not hasattr(self, 'tablewidth'):
            return
        if x > self.tablewidth+w:
            return
        #if event x is within x pixels of divider, draw resize symbol
        nearest = self.within(x, self.table.col_positions, 4)

        if x != x_start and nearest != None:
            #col = self.table.get_col_clicked(event)
            col = self.table.col_positions.index(nearest)-1
            self.nearestcol = col
            #print (nearest,col,self.model.df.columns[col])
            if col == None:
                return
            self.draw_resize_symbol(col)
            self.atdivider = 1
        else:
            self.atdivider = 0
        return

    def handle_right_release(self, event):
        self.rightmenu.destroy()
        return

    def handle_left_shift_click(self, event):
        """Handle shift click, for selecting multiple cols"""

        self.table.delete('colrect')
        self.delete('rect')
        currcol = self.table.currentcol
        colclicked = self.table.get_col_clicked(event)
        if colclicked > currcol:
            self.table.multiplecollist = list(range(currcol, colclicked+1))
        elif colclicked < currcol:
            self.table.multiplecollist = list(range(colclicked, currcol+1))
        else:
            return
        for c in self.table.multiplecollist:
            self.drawRect(c, delete=0)
            self.table.drawSelectedCol(c, delete=0)
        self.table.drawMultipleCells()
        return

    def handle_left_ctrl_click(self, event):
        """Handle ctrl clicks - for multiple column selections"""

        currcol = self.table.currentcol
        colclicked = self.table.get_col_clicked(event)
        multicollist = self.table.multiplecollist
        if 0 <= colclicked < self.table.cols:
            if colclicked not in multicollist:
                multicollist.append(colclicked)
            else:
                multicollist.remove(colclicked)
        self.table.delete('colrect')
        self.delete('rect')
        for c in self.table.multiplecollist:
            self.drawRect(c, delete=0)
            self.table.drawSelectedCol(c, delete=0)
        self.table.drawMultipleCells()
        return

    def handle_double_click(self, event):
        """Double click sorts by this column. """

        colclicked = self.table.get_col_clicked(event)
        if self.sort_ascending == 1:
            self.sort_ascending = 0
        else:
            self.sort_ascending = 1
        self.table.sortTable(ascending=self.sort_ascending)
        return

    def popupMenu(self, event):
        """Add left and right click behaviour for column header"""

        df = self.table.model.df
        if len(df.columns)==0:
            return
        ismulti = util.check_multiindex(df.columns)
        colname = str(df.columns[self.table.currentcol])
        currcol = self.table.currentcol
        multicols = self.table.multiplecollist
        colnames = list(df.columns[multicols])[:4]
        colnames = [str(i)[:20] for i in colnames]
        if len(colnames)>2:
            colnames = ','.join(colnames[:2])+'+%s others' %str(len(colnames)-2)
        else:
            colnames = ','.join(colnames)
        popupmenu = Menu(self, tearoff = 0)
        def popupFocusOut(event):
            popupmenu.unpost()

        columncommands = {"Rename": self.renameColumn,
                          "Add": self.table.addColumn,
                         }
        formatcommands = {

                         }
        popupmenu.add_command(label="Sort by " + colnames + ' \u2193',
                    command=lambda : self.table.sortTable(ascending=[1 for i in multicols]))
        popupmenu.add_command(label="Sort by " + colnames + ' \u2191',
            command=lambda : self.table.sortTable(ascending=[0 for i in multicols]))
        popupmenu.add_command(label="Delete Column(s)", command=self.table.deleteColumn)
        if ismulti == True:
            popupmenu.add_command(label="Flatten Index", command=self.table.flattenIndex)
        popupmenu.add_command(label="Value Counts", command=self.table.valueCounts)
        popupmenu.add_command(label="Set Data Type", command=self.table.setColumnType)

        createSubMenu(popupmenu, 'Column', columncommands)
        popupmenu.bind("<FocusOut>", popupFocusOut)
        popupmenu.focus_set()
        popupmenu.post(event.x_root, event.y_root)
        applyStyle(popupmenu)
        return popupmenu

    def renameColumn(self):
        """Rename column"""

        col = self.table.currentcol
        df = self.model.df
        name = df.columns[col]
        new = simpledialog.askstring("New column name?", "Enter new name:",
                                     initialvalue=name)
        if new != None:
            if new == '':
                messagebox.showwarning("Error", "Name should not be blank.")
                return
            else:
                df.rename(columns={df.columns[col]: new}, inplace=True)
                self.table.tableChanged()
                self.redraw()
        return

    def draw_resize_symbol(self, col):
        """Draw a symbol to show that col can be resized when mouse here"""

        self.delete('resizesymbol')
        w=self.table.cellwidth
        h=25
        wdth=1
        hfac1=0.2
        hfac2=0.4
        x_start=self.table.x_start
        x1,y1,x2,y2 = self.table.getCellCoords(0,col)
        self.create_polygon(x2-3,h/4, x2-10,h/2, x2-3,h*3/4, tag='resizesymbol',
            fill='white', outline='gray', width=wdth)
        self.create_polygon(x2+2,h/4, x2+10,h/2, x2+2,h*3/4, tag='resizesymbol',
            fill='white', outline='gray', width=wdth)
        return

    def drawRect(self,col, tag=None, color=None, outline=None, delete=1):
        """User has clicked to select a col"""

        if tag == None:
            tag = 'rect'
        if color == None:
            color = self.colselectedcolor
        if outline == None:
            outline = 'gray25'
        if delete == 1:
            self.delete(tag)
        w=1
        x1,y1,x2,y2 = self.table.getCellCoords(0,col)
        rect = self.create_rectangle(x1,y1-w,x2,self.height,
                                  fill=color,
                                  outline=outline,
                                  width=w,
                                  tag=tag)
        self.lower(tag)
        return

class RowHeader(Canvas):
    """Class that displays the row headings (or DataFrame index).
       Takes it's size and rendering from the parent table.
       This also handles row/record selection as opposed to cell
       selection"""

    def __init__(self, parent=None, table=None, width=50, bg='gray75'):
        Canvas.__init__(self, parent, bg=bg, width=width, height=None)
        if table != None:
            self.table = table
            self.width = width
            self.inset = 1
            self.textcolor = 'black'
            self.bgcolor = bg
            self.showindex = False
            self.maxwidth = 500
            self.config(height = self.table.height)
            self.startrow = self.endrow = None
            self.model = self.table.model
            self.bind('<Button-1>',self.handle_left_click)
            self.bind("<ButtonRelease-1>", self.handle_left_release)
            self.bind("<Control-Button-1>", self.handle_left_ctrl_click)

            if self.table.ostyp == 'darwin':
                # For mac we bind Shift, left-click to right click
                self.bind("<Button-2>", self.handle_right_click)
                self.bind('<Shift-Button-1>', self.handle_right_click)
            else:
                self.bind("<Button-3>", self.handle_right_click)
            self.bind('<B1-Motion>', self.handle_mouse_drag)
            self.bind('<Shift-Button-1>', self.handle_left_shift_click)
        return

    def redraw(self, align='w', showkeys=False):
        """Redraw row header"""

        self.height = self.table.rowheight * self.table.rows+10
        self.configure(scrollregion=(0,0, self.width, self.height))
        self.delete('rowheader','text')
        self.delete('rect')

        xstart = 1
        pad = 5
        maxw = self.maxwidth
        v = self.table.visiblerows
        if len(v) == 0:
            return
        scale = self.table.getScale()
        h = self.table.rowheight
        index = self.model.df.index
        names = index.names

        if self.table.showindex == True:
            if util.check_multiindex(index) == 1:
                ind = index.values[v]
                cols = [pd.Series(i).astype('object').astype(str)\
                        .replace('nan','') for i in list(zip(*ind))]
                nl = [len(n) if n is not None else 0 for n in names]
                l = [c.str.len().max() for c in cols]
                #pick higher of index names and row data
                l = list(np.maximum(l,nl))
                widths = [i * scale + 6 for i in l]
                xpos = [0]+list(np.cumsum(widths))[:-1]
            else:
                ind = index[v]
                dtype = ind.dtype
                #print (type(ind))
                if type(ind) is pd.CategoricalIndex:
                    ind = ind.astype('str')
                r = ind.fillna('').astype('object').astype('str')
                l = r.str.len().max()
                widths = [l * scale + 6]
                cols = [r]
                xpos = [xstart]
            w = np.sum(widths)
        else:
            rows = [i+1 for i in v]
            cols = [rows]
            l = max([len(str(i)) for i in rows])
            w = l * scale + 6
            widths = [w]
            xpos = [xstart]

        self.widths = widths
        if w>maxw:
            w = maxw
        elif w<45:
            w = 45

        if self.width != w:
            self.config(width=w)
            self.width = w

        i=0
        for col in cols:
            r=v[0]
            x = xpos[i]
            i+=1
            #col=pd.Series(col.tolist()).replace('nan','')
            for row in col:
                text = row
                x1,y1,x2,y2 = self.table.getCellCoords(r,0)
                self.create_rectangle(x,y1,w-1,y2, #fill=self.color,
                                        outline='white', width=1,
                                        tag='rowheader')
                self.create_text(x+pad,y1+h/2, text=text,
                                  fill=self.textcolor, font=self.table.thefont,
                                  tag='text', anchor=align)
                r+=1
        self.config(bg=self.bgcolor)
        return

    def setWidth(self, w):
        """Set width"""
        self.width = w
        self.redraw()
        return

    def clearSelected(self):
        """Clear selected rows"""
        self.delete('rect')
        return

    def handle_left_click(self, event):
        """Handle left click"""

        rowclicked = self.table.get_row_clicked(event)
        self.startrow = rowclicked
        if 0 <= rowclicked < self.table.rows:
            self.delete('rect')
            self.table.delete('entry')
            self.table.delete('multicellrect')
            #set row selected
            self.table.setSelectedRow(rowclicked)
            self.table.drawSelectedRow()
            self.drawSelectedRows(self.table.currentrow)
        return

    def handle_left_release(self,event):
        return

    def handle_left_ctrl_click(self, event):
        """Handle ctrl clicks - for multiple row selections"""

        rowclicked = self.table.get_row_clicked(event)
        multirowlist = self.table.multiplerowlist
        if 0 <= rowclicked < self.table.rows:
            if rowclicked not in multirowlist:
                multirowlist.append(rowclicked)
            else:
                multirowlist.remove(rowclicked)
            self.table.drawMultipleRows(multirowlist)
            self.drawSelectedRows(multirowlist)
        return

    def handle_left_shift_click(self, event):
        """Handle shift click"""

        if self.startrow == None:
            self.startrow = self.table.currentrow
        self.handle_mouse_drag(event)
        return

    def handle_right_click(self, event):
        """respond to a right click"""

        if self.table.enable_menus == False:
            return
        self.delete('tooltip')
        if hasattr(self, 'rightmenu'):
            self.rightmenu.destroy()
        self.rightmenu = self.popupMenu(event, outside=1)
        return

    def handle_mouse_drag(self, event):
        """Handle mouse moved with button held down, multiple selections"""

        if hasattr(self, 'cellentry'):
            self.cellentry.destroy()
        rowover = self.table.get_row_clicked(event)
        colover = self.table.get_col_clicked(event)
        if rowover == None:
            return
        if rowover >= self.table.rows or self.startrow > self.table.rows:
            return
        else:
            self.endrow = rowover
        #draw the selected rows
        if self.endrow != self.startrow:
            if self.endrow < self.startrow:
                rowlist=list(range(self.endrow, self.startrow+1))
            else:
                rowlist=list(range(self.startrow, self.endrow+1))
            self.drawSelectedRows(rowlist)
            self.table.multiplerowlist = rowlist
            self.table.drawMultipleRows(rowlist)
            self.table.drawMultipleCells()
            self.table.allrows = False
        else:
            self.table.multiplerowlist = []
            self.table.multiplerowlist.append(rowover)
            self.drawSelectedRows(rowover)
            self.table.drawMultipleRows(self.table.multiplerowlist)
        return

    def toggleIndex(self):
        """Toggle index display"""

        if self.table.showindex == True:
            self.table.showindex = False
        else:
            self.table.showindex = True
        self.redraw()
        self.table.rowindexheader.redraw()
        return

    def popupMenu(self, event, rows=None, cols=None, outside=None):
        """Add left and right click behaviour for canvas, should not have to override
            this function, it will take its values from defined dicts in constructor"""

        defaultactions = {"Sort by index" : lambda: self.table.sortTable(index=True),
                         "Toggle index" : lambda: self.toggleIndex(),
                         "Select All" : self.table.selectAll,
                         "Add Row(s)" : lambda: self.table.addRows(),
                         "Delete Row(s)" : lambda: self.table.deleteRow(ask=True),
                         "Duplicate Row(s)":  lambda: self.table.duplicateRows()}
        main = ["Sort by index","Toggle index",
                "Add Row(s)","Delete Row(s)", "Duplicate Row(s)"]

        popupmenu = Menu(self, tearoff = 0)
        def popupFocusOut(event):
            popupmenu.unpost()
        for action in main:
            popupmenu.add_command(label=action, command=defaultactions[action])

        popupmenu.bind("<FocusOut>", popupFocusOut)
        popupmenu.focus_set()
        popupmenu.post(event.x_root, event.y_root)
        applyStyle(popupmenu)
        return popupmenu

    def drawSelectedRows(self, rows=None):
        """Draw selected rows, accepts a list or integer"""

        self.delete('rect')
        if type(rows) is not list:
            rowlist=[]
            rowlist.append(rows)
        else:
           rowlist = rows
        for r in rowlist:
            if r not in self.table.visiblerows:
                continue
            self.drawRect(r, delete=0)
        return

    def drawRect(self, row=None, tag=None, color=None, outline=None, delete=1):
        """Draw a rect representing row selection"""

        if tag==None:
            tag='rect'
        if color==None:
            color='#0099CC'
        if outline==None:
            outline='gray25'
        if delete == 1:
            self.delete(tag)
        w=0
        i = self.inset
        x1,y1,x2,y2 = self.table.getCellCoords(row, 0)
        rect = self.create_rectangle(0+i,y1+i,self.width-i,y2,
                                      fill=color,
                                      outline=outline,
                                      width=w,
                                      tag=tag)
        self.lift('text')
        return

class IndexHeader(Canvas):
    """Class that displays the row index headings."""

    def __init__(self, parent=None, table=None, width=40, height=25, bg='gray50'):
        Canvas.__init__(self, parent, bg=bg, width=width, height=height)
        if table != None:
            self.table = table
            self.width = width
            self.height = self.table.rowheight
            self.config(height=self.height)
            self.textcolor = 'white'
            self.bgcolor = bg
            self.startrow = self.endrow = None
            self.model = self.table.model
            self.bind('<Button-1>',self.handle_left_click)
        return

    def redraw(self, align='w'):
        """Redraw row index header"""

        df = self.model.df
        rowheader = self.table.rowheader
        self.width = rowheader.width
        self.delete('text','rect')
        if self.table.showindex == False:
            return
        xstart = 1
        pad = 5
        scale = self.table.getScale()
        h = self.table.rowheight
        self.config(height=h)
        index = df.index
        names = index.names
        if names[0] == None:
            widths = [self.width]
        else:
            widths = rowheader.widths

        if util.check_multiindex(df.columns) == 1:
            levels = df.columns.levels
            h = self.table.rowheight * len(levels)
            y = self.table.rowheight/2 + 2
        else:
            y=2
        i=0; x=1;
        for name in names:
            if name != None:
                w=widths[i]
                self.create_text(x+pad,y+h/2,text=name,
                                    fill=self.textcolor, font=self.table.thefont,
                                    tag='text', anchor=align)
                x=x+widths[i]
                i+=1
        #w=sum(widths)
        self.config(bg=self.bgcolor)
        return

    def handle_left_click(self, event):
        """Handle mouse left mouse click"""
        self.table.selectAll()
        return
    
class Table(Canvas):
    def __init__(self, parent=None, model=None, dataframe=None,
                   width=None, height=None,
                   rows=20, cols=5, showtoolbar=False, showstatusbar=False,
                   editable=True, enable_menus=True,
                   **kwargs):

        Canvas.__init__(self, parent, bg='white', width=width, height=height,relief=GROOVE,scrollregion=(0,0,300,200))
        self.parentframe = parent
        #get platform into a variable
        self.ostyp = util.checkOS()
        self.platform = platform.system()
        self.width = width
        self.height = height
        self.filename = None
        self.showstatusbar = showstatusbar
        self.set_defaults()
        self.currentpage = None
        self.navFrame = None
        self.currentrow = 0
        self.currentcol = 0
        self.reverseorder = 0
        self.startrow = self.endrow = None
        self.startcol = self.endcol = None
        self.allrows = False
        self.multiplerowlist=[]
        self.multiplecollist=[]
        self.col_positions=[]
        self.mode = 'normal'
        self.editable = editable
        self.enable_menus = enable_menus
        self.filtered = False
        self.child = None
        self.queryrow = 4
        self.childrow = 5
        self.currentdir = os.path.expanduser('~')
        self.loadPrefs()
        self.setFont()
        #set any options passed in kwargs to overwrite defaults and prefs
        for key in kwargs:
            self.__dict__[key] = kwargs[key]

        if dataframe is not None:
            self.model = TableModel(dataframe=dataframe)
        elif model != None:
            self.model = model
        else:
            self.model = TableModel(rows=rows,columns=cols)

        self.rows = self.model.getRowCount()
        self.cols = self.model.getColumnCount()
        self.tablewidth = (self.cellwidth)*self.cols
        self.doBindings()
        self.parentframe.bind("<Destroy>", self.close)

        #column specific actions, define for every column type in the model
        #when you add a column type you should edit this dict
        self.columnactions = {'text' : {"Edit":  'drawCellEntry' },
                              'number' : {"Edit": 'drawCellEntry' }}
        #self.setFontSize()
        self.plotted = False
        self.importpath = None
        self.prevdf = None
        return

    def close(self, evt=None):
        if hasattr(self, 'parenttable'):
            return
        if hasattr(self, 'pf') and self.pf is not None:
            #print (self.pf)
            self.pf.close()
        if util.SCRATCH is not None:
            util.SCRATCH.destroy()
            util.SCRATCH = None
        return

    def set_defaults(self):
        """Set default settings"""

        self.cellwidth = 60
        self.maxcellwidth = 300
        self.mincellwidth = 30
        self.rowheight = 20
        self.horizlines = 1
        self.vertlines = 1
        self.autoresizecols = 1
        self.inset = 2
        self.x_start = 0
        self.y_start = 1
        self.linewidth = 1.0
        self.font = 'Arial'
        self.fontsize = 12
        self.fontstyle = ''
        #self.thefont = ('Arial',12)
        self.textcolor = 'black'
        self.cellbackgr = '#F4F4F3'
        self.entrybackgr = 'white'
        self.grid_color = '#ABB1AD'
        self.rowselectedcolor = '#E4DED4'
        self.multipleselectioncolor = '#E0F2F7'
        self.boxoutlinecolor = '#084B8A'
        self.colselectedcolor = '#e4e3e4'
        self.colheadercolor = 'gray25'
        #self.rowheadercolor = 'gray75'
        self.floatprecision = 0
        self.thousandseparator = ''
        self.showindex = False
        self.columnwidths = {}
        self.columncolors = {}
        #store general per column formatting as sub dicts
        self.columnformats = {}
        self.columnformats['alignment'] = {}
        self.rowcolors = pd.DataFrame()
        self.highlighted = None
        #self.bg = Style().lookup('TLabel.label', 'background')
        return

    def setFont(self):
        """Set font tuple"""

        if type(self.fontsize) is str:
            self.fontsize = int(float(self.fontsize))
        if hasattr(self, 'font'):
            self.thefont = (self.font, self.fontsize, self.fontstyle)
        #print (self.thefont)
        return

    def setPrecision(self, x, p):
        """Set precision of a float value"""

        if not pd.isnull(x):
            if x<1:
                x = '{:.{}g}'.format(x, p)
            elif self.thousandseparator == ',':
                x = '{:,.{}f}'.format(x, p)
            else:
                x = '{:.{}f}'.format(x, p)
        return x

    def mouse_wheel(self, event):
        """Handle mouse wheel scroll for windows"""

        if event.num == 5 or event.delta == -120:
            event.widget.yview_scroll(1, UNITS)
            self.rowheader.yview_scroll(1, UNITS)
        if event.num == 4 or event.delta == 120:
            if self.canvasy(0) < 0:
                return
            event.widget.yview_scroll(-1, UNITS)
            self.rowheader.yview_scroll(-1, UNITS)
        self.redrawVisible()
        return

    def doBindings(self):
        """Bind keys and mouse clicks, this can be overriden"""

        self.bind("<Button-1>",self.handle_left_click)
        self.bind("<Double-Button-1>",self.handle_double_click)
        self.bind("<Control-Button-1>", self.handle_left_ctrl_click)
        self.bind("<Shift-Button-1>", self.handle_left_shift_click)

        self.bind("<ButtonRelease-1>", self.handle_left_release)
        if self.ostyp=='darwin':
            #For mac we bind Shift, left-click to right click
            self.bind("<Button-2>", self.handle_right_click)
            self.bind('<Shift-Button-1>',self.handle_right_click)
        else:
            self.bind("<Button-3>", self.handle_right_click)

        self.bind('<B1-Motion>', self.handle_mouse_drag)
        #self.bind('<Motion>', self.handle_motion)

        #self.bind("<Control-x>", self.deleteRow)
        self.bind("<Delete>", self.clearData)
        self.bind("<Control-a>", self.selectAll)

        self.bind("<Right>", self.handle_arrow_keys)
        self.bind("<Left>", self.handle_arrow_keys)
        self.bind("<Up>", self.handle_arrow_keys)
        self.bind("<Down>", self.handle_arrow_keys)
        self.parentframe.master.bind_all("<KP_8>", self.handle_arrow_keys)
        self.parentframe.master.bind_all("<Return>", self.handle_arrow_keys)
        self.parentframe.master.bind_all("<Tab>", self.handle_arrow_keys)
        #if 'windows' in self.platform:
        self.bind("<MouseWheel>", self.mouse_wheel)
        self.bind('<Button-4>', self.mouse_wheel)
        self.bind('<Button-5>', self.mouse_wheel)
        self.focus_set()
        return

    def show(self, callback=None):
        #Add the table and header to the frame
        self.rowheader = RowHeader(self.parentframe, self)
        self.colheader = ColumnHeader(self.parentframe, self, bg='gray25')
        self.rowindexheader = IndexHeader(self.parentframe, self, bg='gray75')
        self.Yscrollbar = AutoScrollbar(self.parentframe,orient=VERTICAL,command=self.set_yviews)
        self.Yscrollbar.grid(row=1,column=2,rowspan=1,sticky='news',pady=0,ipady=0)
        self.Xscrollbar = AutoScrollbar(self.parentframe,orient=HORIZONTAL,command=self.set_xviews)
        self.Xscrollbar.grid(row=2,column=1,columnspan=1,sticky='news')
        self['xscrollcommand'] = self.Xscrollbar.set
        self['yscrollcommand'] = self.Yscrollbar.set
        self.colheader['xscrollcommand'] = self.Xscrollbar.set
        self.rowheader['yscrollcommand'] = self.Yscrollbar.set
        self.parentframe.rowconfigure(1,weight=1)
        self.parentframe.columnconfigure(1,weight=1)

        self.rowindexheader.grid(row=0,column=0,rowspan=1,sticky='news')
        self.colheader.grid(row=0,column=1,rowspan=1,sticky='news')
        self.rowheader.grid(row=1,column=0,rowspan=1,sticky='news')
        self.grid(row=1,column=1,rowspan=1,sticky='news',pady=0,ipady=0)

        self.adjustColumnWidths()
        #bind redraw to resize, may trigger redraws when widgets added
        self.parentframe.bind("<Configure>", self.resized) #self.redrawVisible)
        self.colheader.xview("moveto", 0)
        self.xview("moveto", 0)
        if self.showstatusbar == True:
            self.statusbar = statusBar(self.parentframe, self)
            self.statusbar.grid(row=3,column=0,columnspan=2,sticky='ew')
        #self.redraw(callback=callback)
        self.currwidth = self.parentframe.winfo_width()
        self.currheight = self.parentframe.winfo_height()
        if hasattr(self, 'pf'):
            self.pf.updateData()
        return

    def resized(self, event):
        """Check if size changed when event triggered to avoid unnecessary redraws"""

        if not hasattr(self, 'currwidth'):
            return
        if self.currwidth != self.parentframe.winfo_width() or \
           self.currheight != self.parentframe.winfo_height():
            self.redrawVisible()
        self.currwidth = self.parentframe.winfo_width()
        self.currheight = self.parentframe.winfo_height()

    def remove(self):
        """Close table frame"""

        if hasattr(self, 'parenttable'):
            self.parenttable.child.destroy()
            self.parenttable.child = None
            self.parenttable.plotted = 'main'
        self.parentframe.destroy()
        return

    def getVisibleRegion(self):
        """Get visible region of canvas"""

        x1, y1 = self.canvasx(0), self.canvasy(0)
        #w, h = self.winfo_width(), self.winfo_height()
        #if w <= 1.0 or h <= 1.0:
        w, h = self.master.winfo_width(), self.master.winfo_height()
        x2, y2 = self.canvasx(w), self.canvasy(h)
        return x1, y1, x2, y2

    def getRowPosition(self, y):
        """Set row position"""

        h = self.rowheight
        y_start = self.y_start
        row = (int(y)-y_start)/h
        if row < 0:
            return 0
        if row > self.rows:
            row = self.rows
        return int(row)

    def getColPosition(self, x):
        """Get column position at coord"""

        x_start = self.x_start
        w = self.cellwidth
        i=0
        col=0
        for c in self.col_positions:
            col = i
            if c+w>=x:
                break
            i+=1
        return int(col)

    def getVisibleRows(self, y1, y2):
        """Get the visible row range"""

        start = self.getRowPosition(y1)
        end = self.getRowPosition(y2)+1
        if end > self.rows:
            end = self.rows
        return start, end

    def getVisibleCols(self, x1, x2):
        """Get the visible column range"""

        start = self.getColPosition(x1)
        end = self.getColPosition(x2)+1
        if end > self.cols:
            end = self.cols
        return start, end

    def redrawVisible(self, event=None, callback=None):
        if not hasattr(self, 'colheader'):
            return
        model = self.model
        self.rows = len(self.model.df.index)
        self.cols = len(self.model.df.columns)
        if self.cols == 0 or self.rows == 0:
            self.delete('entry')
            self.delete('rowrect','colrect')
            self.delete('currentrect','fillrect')
            self.delete('gridline','text')
            self.delete('multicellrect','multiplesel')
            self.delete('colorrect')
            self.setColPositions()
            if self.cols == 0:
                self.colheader.redraw()
            if self.rows == 0:
                self.visiblerows = []
                self.rowheader.redraw()
            return
        self.tablewidth = (self.cellwidth) * self.cols
        self.configure(bg=self.cellbackgr)
        self.setColPositions()

        #are we drawing a filtered subset of the recs?
        if self.filtered == True:
            self.delete('colrect')

        self.rowrange = list(range(0,self.rows))
        self.configure(scrollregion=(0,0, self.tablewidth+self.x_start,
                        self.rowheight*self.rows+10))

        x1, y1, x2, y2 = self.getVisibleRegion()
        startvisiblerow, endvisiblerow = self.getVisibleRows(y1, y2)
        self.visiblerows = list(range(startvisiblerow, endvisiblerow))
        startvisiblecol, endvisiblecol = self.getVisibleCols(x1, x2)
        self.visiblecols = list(range(startvisiblecol, endvisiblecol))

        self.drawGrid(startvisiblerow, endvisiblerow)
        align = self.align
        self.delete('fillrect')
        bgcolor = self.cellbackgr
        df = self.model.df

        prec = self.floatprecision
        rows = self.visiblerows
        for col in self.visiblecols:
            coldata = df.iloc[rows,col]
            colname = df.columns[col]
            cfa = self.columnformats['alignment']
            if colname in cfa:
                align = cfa[colname]
            else:
                align = self.align
            if prec != 0:
                if coldata.dtype == 'float64':
                    coldata = coldata.apply(lambda x: self.setPrecision(x, prec), 1)
            coldata = coldata.astype(object).fillna('')
            offset = rows[0]
            for row in self.visiblerows:
                text = coldata.iloc[row-offset]
                self.drawText(row, col, text, align=align)

        self.colorColumns()
        self.colorRows()
        self.colheader.redraw(align=self.align)
        self.rowheader.redraw()
        self.rowindexheader.redraw()
        self.drawSelectedRow()
        self.drawSelectedRect(self.currentrow, self.currentcol)

        if len(self.multiplerowlist)>1:
            self.rowheader.drawSelectedRows(self.multiplerowlist)
            self.drawMultipleRows(self.multiplerowlist)
            self.drawMultipleCells()
        return

    def redraw(self, event=None, callback=None):

        self.redrawVisible(event, callback)
        if hasattr(self, 'statusbar'):
            self.statusbar.update()
        return

    def colorColumns(self, cols=None, color='gray'):
        """Color visible columns"""

        if cols is None:
            cols = self.visiblecols
        self.delete('colorrect')
        for c in cols:
            colname = self.model.df.columns[c]
            if colname in self.columncolors:
                clr = self.columncolors[colname]
                self.drawSelectedCol(c, delete=0, color=clr, tag='colorrect')
        return

    def colorRows(self):
        """Color individual cells in column(s). Requires that the rowcolors
         dataframe has been set. This needs to be updated if the index is reset"""

        #print (self.rowcolors)
        df = self.model.df
        rc = self.rowcolors
        rows = self.visiblerows
        offset = rows[0]
        idx = df.index[rows]
        #diff = df.index.difference(rc.index)
        #print (diff)
        #print (rc)
        for col in self.visiblecols:
            colname = df.columns[col]
            if colname in list(rc.columns):
                colors = rc[colname].loc[idx]
                for row in rows:
                    clr = colors.iloc[row-offset]
                    if not pd.isnull(clr):
                        self.drawRect(row, col, color=clr, tag='colorrect', delete=0)
        return

    def getScale(self):
        try:
            fontsize = self.thefont[1]
        except:
            fontsize = self.fontsize
        scale = 8.5 * float(fontsize)/9
        return scale

    def zoomIn(self):
        self.fontsize = self.fontsize+1
        self.rowheight += 2
        self.colheader.height +=1
        self.setFont()
        self.adjustColumnWidths()
        self.redraw()
        return

    def zoomOut(self):
        self.fontsize = self.fontsize-1
        self.rowheight -= 2
        self.colheader.height -=1
        self.setFont()
        self.adjustColumnWidths()
        self.redraw()
        return

    def expandColumns(self, factor=10):
        """Reduce column widths"""

        self.cellwidth += factor
        widths = self.columnwidths
        for c in widths:
            widths[c] += factor
        self.redraw()
        return

    def contractColumns(self, factor=10):
        """Reduce column widths"""

        self.cellwidth -= factor
        widths = self.columnwidths
        for c in widths:
            widths[c] -= factor
        self.redraw()
        return

    def adjustColumnWidths(self, limit=30):
        fontsize = self.fontsize
        scale = self.getScale()
        if self.cols > limit:
            return
        self.cols = self.model.getColumnCount()
        for col in range(self.cols):
            colname = self.model.getColumnName(col)
            if colname in self.columnwidths:
                w = self.columnwidths[colname]
                #don't adjust large columns as user has probably resized them
                if w>200:
                    continue
            else:
                w = self.cellwidth
            l = self.model.getlongestEntry(col)
            txt = ''.join(['X' for i in range(l+1)])
            tw,tl = util.getTextLength(txt, self.maxcellwidth,
                                       font=self.thefont)
            #print (col,txt,l,tw)
            if tw >= self.maxcellwidth:
                tw = self.maxcellwidth
            elif tw < self.cellwidth:
                tw = self.cellwidth
            self.columnwidths[colname] = tw
        return

    def autoResizeColumns(self):
        """Automatically set nice column widths and draw"""

        self.adjustColumnWidths()
        self.redraw()
        return

    def setColPositions(self):

        df = self.model.df
        self.col_positions=[]
        w = self.cellwidth
        x_pos = self.x_start
        self.col_positions.append(x_pos)
        for col in range(self.cols):
            try:
                colname = df.columns[col].encode('utf-8','ignore').decode('utf-8')
            except:
                colname = str(df.columns[col])
            if colname in self.columnwidths:
                x_pos = x_pos+self.columnwidths[colname]
            else:
                x_pos = x_pos+w
            self.col_positions.append(x_pos)
        self.tablewidth = self.col_positions[len(self.col_positions)-1]
        return

    def sortTable(self, columnIndex=None, ascending=1, index=False):

        df = self.model.df
        if columnIndex == None:
            columnIndex = self.multiplecollist
        if isinstance(columnIndex, int):
            columnIndex = [columnIndex]

        if index == True:
            df.sort_index(inplace=True)
        else:
            colnames = list(df.columns[columnIndex])
            try:
                df.sort_values(by=colnames, inplace=True, ascending=ascending)
            except Exception as e:
                print('could not sort')
                #logging.error("Exception occurred", exc_info=True)
        self.redraw()
        return

    def showIndex(self):
        """Show the row index"""

        self.showindex = True
        return

    def set_rowcolors_index(self):

        df = self.model.df
        self.rowcolors.set_index(df.index, inplace=True)

    def update_rowcolors(self):
        """Update row colors if present so that it syncs with current dataframe."""

        df = self.model.df
        rc = self.rowcolors
        if len(df) == len(self.rowcolors):
            rc.set_index(df.index, inplace=True)
        elif len(df)>len(rc):
            idx = df.index.difference(rc.index)
            #self.rowcolors = rc.append(pd.DataFrame(index=idx))
            self.rowcolors = pd.concat([rc,pd.DataFrame(index=idx)])
        else:
            idx = rc.index.difference(df.index)
            rc.drop(idx,inplace=True)
        #check columns
        cols = list(rc.columns.difference(df.columns))
        if len(cols)>0:
            rc.drop(cols,1,inplace=True)
        cols = list(df.columns.difference(rc.columns))
        if len(cols)>0:
            for col in cols:
                rc[col] = np.nan
        return

    def set_xviews(self,*args):
        """Set the xview of table and col header"""

        self.xview(*args)
        self.colheader.xview(*args)
        self.redrawVisible()
        return

    def set_yviews(self,*args):
        """Set the xview of table and row header"""

        self.yview(*args)
        self.rowheader.yview(*args)
        self.redrawVisible()
        return

    def insertRow(self):
        """Insert a new row"""

        row = self.getSelectedRow()
        key = self.model.insertRow(row)
        #self.rowcolors.append(pd.DataFrame(np.nan,index=[key],columns=self.model.df.columns))
        self.update_rowcolors()
        self.redraw()
        self.tableChanged()
        return

    def addRows(self, num=None):
        """Add new rows"""

        if num == None:
            num = simpledialog.askinteger("Now many rows?",
                                            "Number of rows:",initialvalue=1,
                                             parent=self.parentframe)
        if not num:
            return
        self.storeCurrent()
        keys = self.model.autoAddRows(num)
        self.update_rowcolors()
        self.redraw()
        self.tableChanged()
        return

    def addColumn(self, newname=None):
        """Add a new column"""

        if newname == None:
            coltypes = ['object','float64']
            d = MultipleValDialog(title='New Column',
                                    initialvalues=(coltypes, ''),
                                    labels=('Column Type','Name'),
                                    types=('combobox','string'),
                                    parent = self.parentframe)
            if d.result == None:
                return
            else:
                dtype = d.results[0]
                newname = d.results[1]

        df = self.model.df
        if newname != None:
            if newname in self.model.df.columns:
                messagebox.showwarning("Name exists",
                                        "Name already exists!",
                                        parent=self.parentframe)
            else:
                self.storeCurrent()
                self.model.addColumn(newname, dtype)
                self.parentframe.configure(width=self.width)
                self.update_rowcolors()
                self.redraw()
                self.tableChanged()
        return

    def deleteRow(self, ask=False):
        """Delete a selected row"""

        n = True
        if ask == True:
            n = messagebox.askyesno("Delete",
                                   "Delete selected rows?",
                                    parent=self.parentframe)

        if len(self.multiplerowlist)>1:
            if n == True:
                self.storeCurrent()
                rows = self.multiplerowlist
                self.model.deleteRows(rows)
                self.setSelectedRow(0)
                self.clearSelected()
                self.update_rowcolors()
                self.redraw()
        else:
            if n:
                self.storeCurrent()
                row = self.getSelectedRow()
                self.model.deleteRows([row])
                self.setSelectedRow(row-1)
                self.clearSelected()
                self.update_rowcolors()
                self.redraw()
        return

    def duplicateRows(self):
        """Make copy of rows"""

        rows = self.multiplerowlist
        df = self.model.df
        d = df.iloc[rows]
        self.model.df = pd.concat([df, d])
        self.redraw()
        return

    def deleteColumn(self, ask=True):
        """Delete currently selected column(s)"""

        n = True
        if ask == True:
            n =  messagebox.askyesno("Delete",
                                    "Delete Column(s)?",
                                     parent=self.parentframe)
        if not n:
            return
        self.storeCurrent()
        cols = self.multiplecollist
        self.model.deleteColumns(cols)
        self.setSelectedCol(0)
        self.update_rowcolors()
        self.redraw()
        #self.drawSelectedCol()
        self.tableChanged()
        return

    def tableChanged(self):
        """Callback to be used when dataframe changes so that other
            widgets and data can be updated"""

        self.updateFunctions()
        self.updateWidgets()
        if hasattr(self, 'pf'):
            self.pf.updateData()
        return

    def storeCurrent(self):
        """Store current version of the table before a major change is made"""

        self.prevdf = self.model.df.copy()
        return

    def deleteCells(self, rows, cols, answer=None):
        """Clear the cell contents"""

        if self.editable == False:
            return
        if answer == None:
            answer =  messagebox.askyesno("Clear Confirm",
                                    "Clear this data?",
                                    parent=self.parentframe)
        if not answer:
            return
        self.storeCurrent()
        self.model.deleteCells(rows, cols)
        self.redraw()
        return

    def clearData(self, evt=None):
        """Delete cells from gui event"""

        if self.allrows == True:
            self.deleteColumn()
            return
        rows = self.multiplerowlist
        cols = self.multiplecollist
        self.deleteCells(rows, cols)
        return

    def setColumnType(self):
        """Change the column dtype"""

        df = self.model.df
        col = df.columns[self.currentcol]
        coltypes = ['object','str','int','float64','category']
        curr = df[col].dtype
        d = MultipleValDialog(title='current type is %s' %curr,
                                initialvalues=[coltypes],
                                labels=['Type:'],
                                types=['combobox'],
                                parent = self.parentframe)
        if d.result == None:
            return
        t = d.results[0]
        try:
            self.model.df[col] = df[col].astype(t)
            self.redraw()
        except:
            logging.error("Exception occurred", exc_info=True)
            print('failed')
        return

    def findDuplicates(self):
        """Find duplicate rows"""

        df = self.model.df
        keep = ['first','last']
        d = MultipleValDialog(title='Find duplicates',
                                initialvalues=[False,False,keep],
                                labels=['Remove duplicates:','Use selected columns:','Keep:'],
                                types=['checkbutton','checkbutton','combobox'],
                                parent = self.parentframe)
        if d.result == None:
            return
        remove = d.results[0]
        if d.results[1] is True:
            cols = self.multiplecollist
        else:
            cols = df.columns
        keep = d.results[2]
        new = df[df.duplicated(subset=cols,keep=keep)]
        if remove == True:
            self.model.df = df.drop_duplicates(subset=cols,keep=keep)
            self.redraw()
        if len(new)>0:
            self.createChildTable(new)
        return

    def _getFunction(self, funcname, obj=None):
        if obj != None:
            func = getattr(obj, funcname)
            return func
        if hasattr(pd, funcname):
            func = getattr(pd, funcname)
        elif hasattr(np, funcname):
            func = getattr(np, funcname)
        else:
            return
        return func

    def valueCounts(self):
        """Value counts for column(s)"""

        df = self.model.df
        cols = list(df.columns[self.multiplecollist])
        if len(cols) <2:
            col = cols[0]
            new = df[col].value_counts()
            df = pd.DataFrame(new)
        else:
            #if more than one col we use the first as an index and pivot
            df = df.pivot_table(index=cols[0], columns=cols[1:], aggfunc='size', fill_value=0).T
        self.createChildTable(df, index=True)
        return

    def showAll(self):
        """Re-show unfiltered"""

        if hasattr(self, 'dataframe'):
            self.model.df = self.dataframe
        self.filtered = False
        self.redraw()
        return

    def getRowsFromIndex(self, idx=None):
        """Get row positions from index values"""

        df = self.model.df
        if idx is not None:
            return [df.index.get_loc(i) for i in idx]
        return []

    def updateWidgets(self):
        """Update some dialogs when table changed"""

        if hasattr(self, 'qframe') and self.qframe != None:
            self.qframe.update()
        return

    def _eval(self, df, ex):
        """Evaluate an expression using numexpr"""

        #uses assignments to globals() - check this is ok
        import numexpr as ne
        for c in df:
            globals()[c] = df[c].to_numpy()
        a = ne.evaluate(ex)
        return a

    def evalFunction(self, evt=None):
        """Apply a function to create new columns"""

        #self.convertNumeric(ask=False)
        s = self.evalvar.get()

        if s=='':
            return
        df = self.model.df
        vals = s.split('=')
        if len(vals)==1:
            ex = vals[0]
            n = ex
        else:
            n, ex = vals
        if n == '':
            return
        #evaluate
        try:
            df[n] = self._eval(df, ex)
            self.functionentry.configure(style="White.TCombobox")
        except Exception as e:
            print ('function parse error')
            print (e)
            logging.error("Exception occurred", exc_info=True)
            self.functionentry.configure(style="Red.TCombobox")
            return
        #keep track of which cols are functions?
        self.formulae[n] = ex

        if self.placecolvar.get() == 1:
            cols = df.columns
            self.placeColumn(n,cols[0])
        if self.recalculatevar.get() == 1:
            self.recalculateFunctions(omit=n)
        else:
            self.redraw()
        #update functions list in dropdown
        funclist = ['='.join(i) for i in self.formulae.items()]
        self.functionentry['values'] = funclist
        return

    def recalculateFunctions(self, omit=None):
        """Re evaluate any columns that were derived from functions
        and dependent on other columns (except self derived?)"""

        df = self.model.df
        for n in self.formulae:
            if n==omit: continue
            ex = self.formulae[n]
            #need to check if self calculation here...
            try:
                df[n] = self._eval(df, ex)
            except:
                logging.error("Exception occurred", exc_info=True)
                print('could not calculate %s' %ex)
        self.redraw()
        return

    def updateFunctions(self):
        """Remove functions if a column has been deleted"""

        if not hasattr(self, 'formulae'):
            return
        df = self.model.df
        cols = list(df.columns)
        for n in list(self.formulae.keys()):
            if n not in cols:
                del(self.formulae[n])
        return

    def resizeColumn(self, col, width):
        """Resize a column by dragging"""

        colname = self.model.getColumnName(col)
        if self.colheader.wrap == True:
            if width<40:
                width=40
        self.columnwidths[colname] = width
        self.setColPositions()
        self.delete('colrect')
        #self.drawSelectedCol(self.currentcol)
        self.redraw()
        return

    def get_row_clicked(self, event):
        """Get row where event on canvas occurs"""

        h=self.rowheight
        #get coord on canvas, not window, need this if scrolling
        y = int(self.canvasy(event.y))
        y_start=self.y_start
        rowc = int((int(y)-y_start)/h)
        return rowc

    def get_col_clicked(self,event):
        """Get column where event on the canvas occurs"""

        w = self.cellwidth
        x = int(self.canvasx(event.x))
        x_start = self.x_start
        for colpos in self.col_positions:
            try:
                nextpos = self.col_positions[self.col_positions.index(colpos)+1]
            except:
                nextpos = self.tablewidth
            if x > colpos and x <= nextpos:
                #print 'x=', x, 'colpos', colpos, self.col_positio.drawSelectedRectns.index(colpos)
                return self.col_positions.index(colpos)
        return

    def setSelectedRow(self, row=None):
        """Set currently selected row and reset multiple row list"""

        self.currentrow = row
        self.startrow = row
        self.multiplerowlist = []
        if row != None:
            self.multiplerowlist.append(row)
        return

    def setSelectedCol(self, col):
        """Set currently selected column"""

        self.currentcol = col
        self.multiplecollist = []
        self.multiplecollist.append(col)
        return

    def setSelectedRows(self, rows):

        self.startrow = rows[0]
        self.endrow = rows[-1]
        self.multiplerowlist = []
        for r in rows:
            self.multiplerowlist.append(r)

    def setSelectedCells(self, startrow, endrow, startcol, endcol):
        """Set a block of cells selected"""

        self.currentrow = startrow
        self.currentcol = startcol
        if startrow < 0 or startcol < 0:
            return
        if endrow > self.rows or endcol > self.cols:
            return
        for r in range(startrow, endrow):
            self.multiplerowlist.append(r)
        for c in range(startcol, endcol):
            self.multiplecollist.append(c)
        return

    def getSelectedRow(self):
        """Get currently selected row"""
        return self.currentrow

    def getSelectedColumn(self):
        """Get currently selected column"""
        return self.currentcol

    def selectAll(self, evt=None):
        """Select all rows and cells"""

        self.startrow = 0
        self.endrow = self.rows
        self.multiplerowlist = list(range(self.startrow,self.endrow))
        self.drawMultipleRows(self.multiplerowlist)
        self.startcol = 0
        self.endcol = self.cols
        self.multiplecollist = list(range(self.startcol, self.endcol))
        self.drawMultipleCells()
        return

    def selectNone(self):
        """Deselect current, called when table is redrawn with
        completely new cols and rows e.g. after model is updated."""

        self.multiplecollist = []
        self.multiplerowlist = []
        self.startrow = self.endrow = 0
        self.delete('multicellrect','multiplesel','colrect')
        return

    def getCellCoords(self, row, col):
        """Get x-y coordinates to drawing a cell in a given row/col"""

        colname=self.model.getColumnName(col)
        if colname in self.columnwidths:
            w=self.columnwidths[colname]
        else:
            w=self.cellwidth
        h=self.rowheight
        x_start=self.x_start
        y_start=self.y_start

        #get nearest rect co-ords for that row/col
        x1=self.col_positions[col]
        y1=y_start+h*row
        x2=x1+w
        y2=y1+h
        return x1,y1,x2,y2

    def getCanvasPos(self, row, col):
        """Get the cell x-y coords as a fraction of canvas size"""

        if self.rows==0:
            return None, None
        x1,y1,x2,y2 = self.getCellCoords(row,col)
        cx=float(x1)/self.tablewidth
        cy=float(y1)/(self.rows*self.rowheight)
        return cx, cy

    def isInsideTable(self,x,y):
        """Returns true if x-y coord is inside table bounds"""

        if self.x_start < x < self.tablewidth and self.y_start < y < self.rows*self.rowheight:
            return 1
        else:
            return 0
        return answer

    def setRowHeight(self, h):
        """Set the row height"""
        self.rowheight = h
        return

    def clearSelected(self):
        """Clear selections"""

        self.delete('rect')
        self.delete('entry')
        self.delete('tooltip')
        self.delete('searchrect')
        self.delete('colrect')
        self.delete('multicellrect')
        return

    def handle_left_click(self, event):
        """Respond to a single press"""

        self.clearSelected()
        self.allrows = False
        #which row and column is the click inside?
        rowclicked = self.get_row_clicked(event)
        colclicked = self.get_col_clicked(event)
        if colclicked == None:
            return
        self.focus_set()

        if hasattr(self, 'cellentry'):
            self.cellentry.destroy()
        #ensure popup menus are removed if present
        if hasattr(self, 'rightmenu'):
            self.rightmenu.destroy()
        if hasattr(self.colheader, 'rightmenu'):
            self.colheader.rightmenu.destroy()

        self.startrow = rowclicked
        self.endrow = rowclicked
        self.startcol = colclicked
        self.endcol = colclicked
        #reset multiple selection list
        self.multiplerowlist=[]
        self.multiplerowlist.append(rowclicked)
        if 0 <= rowclicked < self.rows and 0 <= colclicked < self.cols:
            self.setSelectedRow(rowclicked)
            self.setSelectedCol(colclicked)
            self.drawSelectedRect(self.currentrow, self.currentcol)
            self.drawSelectedRow()
            self.rowheader.drawSelectedRows(rowclicked)
            self.colheader.delete('rect')
        if hasattr(self, 'cellentry'):
            self.cellentry.destroy()
        return

    def handle_left_release(self,event):
        """Handle left mouse button release event"""

        self.endrow = self.get_row_clicked(event)
        df = self.model.df
        colname = df.columns[self.currentcol]
        dtype = df.dtypes[colname]

        if dtype.name == 'category':
            #drop down menu for category entry
            row = self.get_row_clicked(event)
            col = self.get_col_clicked(event)
            x1,y1,x2,y2 = self.getCellCoords(row,col)
            self.dropvar = StringVar()
            val = self.model.getValueAt(row,col)
            #get categories
            optionlist = list(df[colname].cat.categories[:50])
            dropmenu = OptionMenu(self, self.dropvar, val, *optionlist)
            self.dropvar.trace('w', self.handleEntryMenu)
            self.create_window(x1,y1,
                                width=120,height=30,
                                window=dropmenu, anchor='nw',
                                tag='entry')
        return

    def handle_left_ctrl_click(self, event):
        """Handle ctrl clicks for multiple row selections"""

        rowclicked = self.get_row_clicked(event)
        colclicked = self.get_col_clicked(event)
        if 0 <= rowclicked < self.rows and 0 <= colclicked < self.cols:
            if rowclicked not in self.multiplerowlist:
                self.multiplerowlist.append(rowclicked)
            else:
                self.multiplerowlist.remove(rowclicked)
            self.drawMultipleRows(self.multiplerowlist)
            if colclicked not in self.multiplecollist:
                self.multiplecollist.append(colclicked)
            self.drawMultipleCells()
        return

    def handle_left_shift_click(self, event):
        """Handle shift click, for selecting multiple rows"""

        self.handle_mouse_drag(event)
        return

    def handle_mouse_drag(self, event):
        """Handle mouse moved with button held down, multiple selections"""

        if hasattr(self, 'cellentry'):
            self.cellentry.destroy()
        rowover = self.get_row_clicked(event)
        colover = self.get_col_clicked(event)
        if colover == None or rowover == None:
            return

        if rowover >= self.rows or self.startrow > self.rows:
            return
        else:
            self.endrow = rowover
        #do columns
        if colover > self.cols or self.startcol > self.cols:
            return
        else:
            self.endcol = colover
            if self.endcol < self.startcol:
                self.multiplecollist=list(range(self.endcol, self.startcol+1))
            else:
                self.multiplecollist=list(range(self.startcol, self.endcol+1))
            #print self.multiplecollist
        #draw the selected rows
        if self.endrow != self.startrow:
            if self.endrow < self.startrow:
                self.multiplerowlist=list(range(self.endrow, self.startrow+1))
            else:
                self.multiplerowlist=list(range(self.startrow, self.endrow+1))
            self.drawMultipleRows(self.multiplerowlist)
            self.rowheader.drawSelectedRows(self.multiplerowlist)
            #draw selected cells outline using row and col lists
            self.drawMultipleCells()
        else:
            self.multiplerowlist = []
            self.multiplerowlist.append(self.currentrow)
            if len(self.multiplecollist) >= 1:
                self.drawMultipleCells()
            self.delete('multiplesel')
        return

    def handle_arrow_keys(self, event):
        """Handle arrow keys press"""
        #print event.keysym

        row = self.get_row_clicked(event)
        col = self.get_col_clicked(event)
        x,y = self.getCanvasPos(self.currentrow, self.currentcol-1)
        rmin = self.visiblerows[0]
        rmax = self.visiblerows[-1]-2
        cmax = self.visiblecols[-1]-1
        cmin = self.visiblecols[0]
        if x == None:
            return

        if event.keysym == 'Up':
            if self.currentrow == 0:
                return
            else:
                #self.yview('moveto', y)
                #self.rowheader.yview('moveto', y)
                self.currentrow  = self.currentrow - 1
        elif event.keysym == 'Down':
            if self.currentrow >= self.rows-1:
                return
            else:
                self.currentrow  = self.currentrow + 1
        elif event.keysym == 'Right' or event.keysym == 'Tab':
            if self.currentcol >= self.cols-1:
                if self.currentrow < self.rows-1:
                    self.currentcol = 0
                    self.currentrow  = self.currentrow + 1
                else:
                    return
            else:
                self.currentcol  = self.currentcol + 1
        elif event.keysym == 'Left':
            if self.currentcol>0:
                self.currentcol = self.currentcol - 1

        if self.currentcol > cmax or self.currentcol <= cmin:
            #print (self.currentcol, self.visiblecols)
            self.xview('moveto', x)
            self.colheader.xview('moveto', x)
            self.redraw()

        if self.currentrow <= rmin:
            #we need to shift y to page up enough
            vh=len(self.visiblerows)/2
            x,y = self.getCanvasPos(self.currentrow-vh, 0)

        if self.currentrow >= rmax or self.currentrow <= rmin:
            self.yview('moveto', y)
            self.rowheader.yview('moveto', y)
            self.redraw()

        self.drawSelectedRect(self.currentrow, self.currentcol)
        coltype = self.model.getColumnType(self.currentcol)
        return

    def handle_double_click(self, event):
        """Do double click stuff. Selected row/cols will already have
           been set with single click binding"""

        row = self.get_row_clicked(event)
        col = self.get_col_clicked(event)
        self.drawCellEntry(self.currentrow, self.currentcol)
        return

    def handle_right_click(self, event):
        """respond to a right click"""

        self.delete('tooltip')
        self.rowheader.clearSelected()
        if hasattr(self, 'rightmenu'):
            self.rightmenu.destroy()
        if self.enable_menus == False:
            return
        rowclicked = self.get_row_clicked(event)
        colclicked = self.get_col_clicked(event)
        if colclicked == None:
            self.rightmenu = self.popupMenu(event, outside=1)
            return

        if (rowclicked in self.multiplerowlist or self.allrows == True) and colclicked in self.multiplecollist:
            self.rightmenu = self.popupMenu(event, rows=self.multiplerowlist, cols=self.multiplecollist)
        else:
            if 0 <= rowclicked < self.rows and 0 <= colclicked < self.cols:
                self.clearSelected()
                self.allrows = False
                self.setSelectedRow(rowclicked)
                self.setSelectedCol(colclicked)
                self.drawSelectedRect(self.currentrow, self.currentcol)
                self.drawSelectedRow()
            if self.isInsideTable(event.x,event.y) == 1:
                self.rightmenu = self.popupMenu(event,rows=self.multiplerowlist, cols=self.multiplecollist)
            else:
                self.rightmenu = self.popupMenu(event, outside=1)
        return

    def placeColumn(self, col1, col2):
        """Move col1 next to col2, useful for placing a new column
        made from the first one next to it so user can see it easily"""

        ind1 = self.model.df.columns.get_loc(col1)
        ind2 = self.model.df.columns.get_loc(col2)
        self.model.moveColumn(ind1, ind2+1)
        self.redraw()
        return

    def gotonextCell(self):
        """Move highlighted cell to next cell in row or a new col"""

        if hasattr(self, 'cellentry'):
            self.cellentry.destroy()
        self.currentrow = self.currentrow+1
        self.drawSelectedRect(self.currentrow, self.currentcol)
        return

    def movetoSelection(self, row=None, col=0, idx=None, offset=0):
        """Move to a specific row/col, updating table"""

        if row is None:
            if idx is None:
                return
            rows = self.getRowsFromIndex(idx)
            row=rows[0]
        self.setSelectedRow(row)
        self.drawSelectedRow()
        x,y = self.getCanvasPos(abs(row-offset), col)
        #print (row,col)
        self.xview('moveto', x)
        self.yview('moveto', y)
        self.colheader.xview('moveto', x)
        self.rowheader.yview('moveto', y)
        self.rowheader.redraw()
        return

    def pivot(self):
        """Pivot table"""

        df = self.model.df
        cols = list(df.columns)
        valcols = list(df.select_dtypes(include=[np.float64,np.int32,np.int64]))
        funcs = ['mean','sum','count','max','min','std','first','last']
        d = MultipleValDialog(title='Pivot',
                                initialvalues=(cols,cols,valcols,funcs),
                                labels=('Index:', 'Columns:', 'Values:','Agg Function:'),
                                types=('combobox','listbox','listbox','combobox'),
                                tooltips=('a unique index to reshape on','column with variables',
                                    'selecting no values uses all remaining cols',
                                    'function to aggregate on'),
                                parent = self.parentframe)
        if d.result == None:
            return
        index = d.results[0]
        column = d.results[1]
        values = d.results[2]
        func = d.results[3]
        if values == '': values = None
        elif len(values) == 1: values = values[0]

        p = pd.pivot_table(df, index=index, columns=column, values=values, aggfunc=func)
        #print (p)
        self.tableChanged()
        if type(p) is pd.Series:
            p = pd.DataFrame(p)
        self.createChildTable(p, 'pivot-%s-%s' %(index,column), index=True)
        return

    def createChildTable(self, df, title=None, index=False, out=False):
        """Add the child table"""

        self.closeChildTable()
        if out == True:
            win = Toplevel()
            x,y,w,h = self.getGeometry(self.master)
            win.geometry('+%s+%s' %(int(x+w/2),int(y+h/2)))
            if title != None:
                win.title(title)
        else:
            win = Frame(self.parentframe)
            win.grid(row=self.childrow,column=0,columnspan=2,sticky='news')
        self.childframe = win
        newtable = self.__class__(win, dataframe=df, showstatusbar=1)
        newtable.parenttable = self
        newtable.adjustColumnWidths()
        newtable.show()
        self.child = newtable
        if hasattr(self, 'pf'):
            newtable.pf = self.pf
        if index==True:
            newtable.showIndex()
        return

    def closeChildTable(self):
        """Close the child table"""

        if self.child != None:
            self.child.destroy()
        if hasattr(self, 'childframe'):
            self.childframe.destroy()
        return

    def showInfo(self):
        """Show dataframe info"""

        df = self.model.df
        import io
        buf = io.StringIO()
        df.info(verbose=True,buf=buf,memory_usage=True)
        from pandastable.dialogs import SimpleEditor
        w = Toplevel(self.parentframe)
        w.grab_set()
        w.transient(self)
        ed = SimpleEditor(w, height=25)
        ed.pack(in_=w, fill=BOTH, expand=Y)
        ed.text.insert(END, buf.getvalue())
        return

    def doExport(self, filename=None):

        if filename == None:
            filename = filedialog.asksaveasfilename(parent=self.master,
                                                      defaultextension='.csv',
                                                      initialdir = os.getcwd(),
                                                      filetypes=[("csv","*.csv"),
                                                           ("excel","*.xls"),
                                                           ("html","*.html"),
                                                        ("All files","*.*")])
        if filename:
            self.model.save(filename)
        return

    def popupMenu(self, event, rows=None, cols=None, outside=None):
        """Add left and right click behaviour for canvas, should not have to override
            this function, it will take its values from defined dicts in constructor"""

        defaultactions = {
                        "Add Row(s)" : lambda: self.addRows(),
                        "Add Column(s)" : lambda: self.addColumn(),
                        "Delete Column(s)" : lambda: self.deleteColumn(),
                        "Clear Data" : lambda: self.deleteCells(rows, cols),
                        "Select All" : self.selectAll,
                        "Table Info" : self.showInfo,
                        "Export": self.doExport,
                        "Preferences" : self.showPreferences}
        main = [ "Fill Down","Clear Data"]
        general = ["Select All", "Table Info", "Preferences", 'Export']
        def add_commands(fieldtype):
            """Add commands to popup menu for column type and specific cell"""
            functions = self.columnactions[fieldtype]
            for f in list(functions.keys()):
                func = getattr(self, functions[f])
                popupmenu.add_command(label=f, command= lambda : func(row,col))
            return

        popupmenu = Menu(self, tearoff = 0)
        def popupFocusOut(event):
            popupmenu.unpost()

        if outside == None:
            #if outside table, just show general items
            row = self.get_row_clicked(event)
            col = self.get_col_clicked(event)
            coltype = self.model.getColumnType(col)
            def add_defaultcommands():
                """now add general actions for all cells"""
                for action in main:
                    if action == 'Fill Down' and (rows == None or len(rows) <= 1):
                        continue
                    if action == 'Fill Right' and (cols == None or len(cols) <= 1):
                        continue
                    else:
                        popupmenu.add_command(label=action, command=defaultactions[action])
                return

            if coltype in self.columnactions:
                add_commands(coltype)
            add_defaultcommands()

        for action in general:
            popupmenu.add_command(label=action, command=defaultactions[action])

        popupmenu.add_separator()
        popupmenu.bind("<FocusOut>", popupFocusOut)
        popupmenu.focus_set()
        popupmenu.post(event.x_root, event.y_root)
        applyStyle(popupmenu)
        return popupmenu

    def drawGrid(self, startrow, endrow):
        """Draw the table grid lines"""

        self.delete('gridline','text')
        rows=len(self.rowrange)
        cols=self.cols
        w = self.cellwidth
        h = self.rowheight
        x_start=self.x_start
        y_start=self.y_start
        x_pos=x_start

        if self.vertlines==1:
            for col in range(cols+1):
                x=self.col_positions[col]
                self.create_line(x,y_start,x,y_start+rows*h, tag='gridline',
                                     fill=self.grid_color, width=self.linewidth)
        if self.horizlines==1:
            for row in range(startrow, endrow+1):
                y_pos=y_start+row*h
                self.create_line(x_start,y_pos,self.tablewidth,y_pos, tag='gridline',
                                    fill=self.grid_color, width=self.linewidth)
        return

    def drawSelectedRect(self, row, col, color=None, fillcolor=None):
        """User has clicked to select a cell"""

        if col >= self.cols:
            return
        self.delete('currentrect')
        if color == None:
            color = 'gray25'
        w=2
        if row == None:
            return
        x1,y1,x2,y2 = self.getCellCoords(row,col)
        rect = self.create_rectangle(x1+w/2+1,y1+w/2+1,x2-w/2,y2-w/2,
                                  outline=color,
                                  fill=fillcolor,
                                  width=w,
                                  tag='currentrect')
        #raise text above all
        self.lift('celltext'+str(col)+'_'+str(row))
        return

    def drawRect(self, row, col, color=None, tag=None, delete=1):
        """Cell is colored"""

        if delete==1:
            self.delete('cellbg'+str(row)+str(col))
        if color==None or color==self.cellbackgr:
            return
        else:
            bg=color
        if tag==None:
            recttag='fillrect'
        else:
            recttag=tag
        w=1
        x1,y1,x2,y2 = self.getCellCoords(row,col)
        rect = self.create_rectangle(x1+w/2,y1+w/2,x2-w/2,y2-w/2,
                                  fill=bg,
                                  outline=bg,
                                  width=w,
                                  tag=(recttag,'cellbg'+str(row)+str(col)))
        self.lower(recttag)
        return

    def handleCellEntry(self, row, col):
        """Callback for cell entry"""

        value = self.cellentryvar.get()
        if self.filtered == 1:
            df = self.dataframe
        else:
            df = None
        self.model.setValueAt(value,row,col,df=df)

        self.drawText(row, col, value, align=self.align)
        self.delete('entry')
        self.gotonextCell()
        return

    def handleEntryMenu(self, *args):
        """Callback for option menu in categorical columns entry"""

        value = self.dropvar.get()
        self.delete('entry')
        row = self.currentrow
        col = self.currentcol
        try:
            self.model.setValueAt(value,row,col)
        except:
            self.model.setValueAt(float(value),row,col)
        self.drawText(row, col, value, align=self.align)
        return

    def drawCellEntry(self, row, col, text=None):
        """When the user single/double clicks on a text/number cell,
          bring up entry window and allow edits."""

        if self.editable == False:
            return
        h = self.rowheight
        model = self.model
        text = self.model.getValueAt(row, col)
        if pd.isnull(text):
            text = ''
        x1,y1,x2,y2 = self.getCellCoords(row,col)
        w=x2-x1
        self.cellentryvar = txtvar = StringVar()
        txtvar.set(text)

        self.cellentry = Entry(self.parentframe,width=20,
                        textvariable=txtvar,
                        takefocus=1,
                        font=self.thefont)
        self.cellentry.icursor(END)
        self.cellentry.bind('<Return>', lambda x: self.handleCellEntry(row,col))
        self.cellentry.focus_set()
        self.entrywin = self.create_window(x1,y1,
                                width=w,height=h,
                                window=self.cellentry,anchor='nw',
                                tag='entry')
        return

    def drawText(self, row, col, celltxt, align=None, single_line=True):
        """Draw the text inside a cell area"""

        self.delete('celltext'+str(col)+'_'+str(row))
        h = self.rowheight
        x1,y1,x2,y2 = self.getCellCoords(row,col)
        w=x2-x1
        wrap = False
        pad=5
        #if type(celltxt) is np.float64:
        #    celltxt = np.round(celltxt,3)
        celltxt = str(celltxt)

        if single_line:
            celltxt = celltxt.strip().split('\n', 1)[0]

        length = len(celltxt)
        if length == 0:
            return

        if w<=10:
            return
        if w < 18:
            celltxt = '.'
            return

        #fgcolor = self.textcolor
        if align == None:
            align = 'center'
        elif align == 'w':
            x1 = x1-w/2+pad
        elif align == 'e':
            x1 = x1+w/2-pad

        tw,newlength = util.getTextLength(celltxt, w-pad, font=self.thefont)
        width=0
        celltxt = celltxt[0:int(newlength)]
        y=y1+h/2
        rect = self.create_text(x1+w/2,y,
                                  text=celltxt,
                                  fill=self.textcolor,
                                  font=self.thefont,
                                  anchor=align,
                                  tag=('text','celltext'+str(col)+'_'+str(row)),
                                  width=width)
        return

    def drawSelectedRow(self):
        """Draw a highlight rect for the currently selected rows"""

        self.delete('rowrect')
        row = self.currentrow

        if row == None:
            return
        x1,y1,x2,y2 = self.getCellCoords(row,0)
        x2 = self.tablewidth
        rect = self.create_rectangle(x1,y1,x2,y2,
                                  fill=self.rowselectedcolor,
                                  outline=self.rowselectedcolor,
                                  tag='rowrect')
        self.lower('rowrect')
        #self.lower('fillrect')
        self.lower('colorrect')
        self.rowheader.drawSelectedRows(self.currentrow)
        return

    def drawSelectedCol(self, col=None, delete=1, color=None, tag='colrect'):
        """Draw a highlight rect for the current column selection"""

        if color == None:
            color = self.colselectedcolor
        if delete == 1:
            self.delete(tag)
        if len(self.model.df.columns) == 0:
            return
        if col == None:
            col = self.currentcol
        w=2
        x1,y1,x2,y2 = self.getCellCoords(0,col)
        y2 = self.rows * self.rowheight
        rect = self.create_rectangle(x1+w/2,y1+w/2,x2,y2+w/2,
                                     width=w,fill=color,outline='',
                                     tag=tag)
        self.lower('rowrect')
        self.lower('colrect')
        return

    def drawMultipleRows(self, rowlist):
        """Draw more than one row selection"""

        self.delete('multiplesel')
        #self.delete('rowrect')
        cols = self.visiblecols
        rows = list(set(rowlist) & set(self.visiblerows))
        if len(rows)==0:
            return
        for col in cols:
            colname = self.model.df.columns[col]
            #if col is colored we darken it
            if colname in self.columncolors:
                clr = self.columncolors[colname]
                clr = util.colorScale(clr, -30)
            else:
                clr = self.rowselectedcolor
            for r in rows:
                x1,y1,x2,y2 = self.getCellCoords(r,col)
                rect = self.create_rectangle(x1,y1,x2,y2,
                                          fill=clr,
                                          outline=self.rowselectedcolor,
                                          tag=('multiplesel','rowrect'))
        self.lower('multiplesel')
        self.lower('fillrect')
        self.lower('colorrect')
        return

    def drawMultipleCols(self):
        """Draw multiple column selections"""

        for c in self.multiplecollist:
            self.drawSelectedCol(c, delete=False)
        return

    def drawMultipleCells(self):
        """Draw an outline box for multiple cell selection"""

        self.delete('currentrect')
        self.delete('multicellrect')
        rows = self.multiplerowlist
        cols = self.multiplecollist
        if len(rows) == 0 or len(cols) == 0:
            return
        w=2
        x1,y1,a,b = self.getCellCoords(rows[0],cols[0])
        c,d,x2,y2 = self.getCellCoords(rows[len(rows)-1],cols[len(cols)-1])
        rect = self.create_rectangle(x1+w/2,y1+w/2,x2,y2,
                             outline=self.boxoutlinecolor, width=w,
                             tag='multicellrect')
        return

    def showPreferences(self):
        """Preferences dialog"""

        options = config.load_options()
        f = config.preferencesDialog(self, options, table=self)
        return

    def loadPrefs(self, prefs=None):
        """Load preferences from defaults"""

        options = config.load_options()
        config.apply_options(options, self)
        return


    def getGeometry(self, frame):
        """Get frame geometry"""
        return frame.winfo_rootx(), frame.winfo_rooty(), frame.winfo_width(), frame.winfo_height()

class statusBar(Frame):
    """Status bar class"""
    def __init__(self, parent=None, parentapp=None):

        Frame.__init__(self, parent)
        self.parentframe = parent
        self.parentapp = parentapp
        df = self.parentapp.model.df
        sfont = ("Helvetica bold", 10)
        clr = '#A10000'
        self.rowsvar = StringVar()
        self.rowsvar.set(len(df))
        l=Label(self,textvariable=self.rowsvar,font=sfont,foreground=clr)
        l.pack(fill=X, side=LEFT)
        Label(self,text='rows x',font=sfont,foreground=clr).pack(side=LEFT)
        self.colsvar = StringVar()
        self.colsvar.set(len(df.columns))
        l=Label(self,textvariable=self.colsvar,font=sfont,foreground=clr)
        l.pack(fill=X, side=LEFT)
        Label(self,text='columns',font=sfont,foreground=clr).pack(side=LEFT)
        self.filenamevar = StringVar()
        l=Label(self,textvariable=self.filenamevar,font=sfont)
        l.pack(fill=X, side=RIGHT)
        fr = Frame(self)
        fr.pack(fill=Y,side=RIGHT)

        img = images.contract_col()
        addButton(fr, 'Contract Cols', self.parentapp.contractColumns, img, 'contract columns', side=LEFT, padding=1)
        img = images.expand_col()
        addButton(fr, 'Expand Cols', self.parentapp.expandColumns, img, 'expand columns', side=LEFT, padding=1)
        img = images.zoom_out()
        addButton(fr, 'Zoom Out', self.parentapp.zoomOut, img, 'zoom out', side=LEFT, padding=1)
        img = images.zoom_in()
        addButton(fr, 'Zoom In', self.parentapp.zoomIn, img, 'zoom in', side=LEFT, padding=1)
        return

    def update(self):
        """Update status bar"""

        model = self.parentapp.model
        self.rowsvar.set(len(model.df))
        self.colsvar.set(len(model.df.columns))
        if self.parentapp.filename != None:
            self.filenamevar.set(self.parentapp.filename)
        return
