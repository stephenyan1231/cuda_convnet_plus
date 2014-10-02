

class HTMLwriter:
    def __init__(self):
        pass
    
    @staticmethod
    def writeBlankLine(fp):
        fp.write('\n')
    
    @staticmethod
    def writeParagraph(fp, tag, text, options=[]):
        fp.write('<%s>' % tag)
        for op in options:
            fp.write('<%s>' % op)
        fp.write(text)
        for op in options:
            fp.write('</%s>' % op)
        fp.write('</%s>\n' % tag)
        
    @staticmethod
    def writeHyperLink(fp, URL):
        fp.write('<a href="%s">' % URL)
    
    @staticmethod
    def insertImage(fp,URL,alt=""):
        fp.write('<img src="%s" alt="%s">' % (URL,alt))
        