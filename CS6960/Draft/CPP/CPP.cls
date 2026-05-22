%--------------------------------------------------------------------
%Created on:    Nov. 03, 2006
%Author:        Tzvetelin Petrov
%--------------------------------------------------------------------
% Modified by Berit Givens May 07, 2014
% Modified by Rafael Morales May 10, 2014
% Modified by Jillian Cannons January 31, 2019; April 16, 2019; 
%                             April 9, 2020; November 16, 2023
%--------------------------------------------------------------------

%--------------------------------------------------------------------
% Register the class.
%--------------------------------------------------------------------
    \NeedsTeXFormat{LaTeX2e} \ProvidesClass{CPP}[2006/11/03 Cal Poly Pomona Grad Study Format]

%--------------------------------------------------------------------
% Load the report class to handle unknown options.
%--------------------------------------------------------------------
    \DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
    \ProcessOptions\relax \LoadClass[letterpaper,12pt]{report}

%--------------------------------------------------------------------
% Define variables.
%--------------------------------------------------------------------
    %Title name line 1.
    \newcommand{\titleone}[1]{\gdef\@titleone{\mbox{\uppercase{#1}}}}

    %Title name line 2.
    \newcommand{\titletwo}[1]{\gdef\@titletwo{\mbox{\uppercase{#1}}}}

    %Title name line 3.
    \newcommand{\titlethree}[1]{\gdef\@titlethree{\mbox{\uppercase{#1}}}}
    
    %The type of this document: Thesis/Project.
    \newcommand{\doctype}[1]{\gdef\@doctype{#1}}
    
    %The type of this document in UPPERCASE: Thesis/Project.
    \newcommand{\doctypeUp}[1]{\gdef\@doctypeUp{\mbox{\uppercase{#1}}}}

    %Full name of the degree: Master of Science.
    \newcommand{\degree}[1]{\gdef\@degree{#1}}
    
    %Full name of the field in which the degree is awarded: Mathematics.
    \newcommand{\field}[1]{\gdef\@field{#1}}    

    %Full name of the author.
    \newcommand{\Author}[1]{\gdef\@Author{#1}}
    
    %Full name of the advisor
    \newcommand{\Advisor}[1]{\gdef\@Advisor{#1}}

    %Full name of the committee member 2
    \newcommand{\MemberA}[1]{\gdef\@MemberA{#1}}

    %Full name of the committee member 3
    \newcommand{\MemberB}[1]{\gdef\@MemberB{#1}}    

    %Date the degree is awarded.
    \newcommand{\Year}[1]{\gdef\@Year{#1}}
    \newcommand{\Semester}[1]{\gdef\@Semester{#1}}
    
    %The acknowledgments text
    \newcommand{\Acknowledgments}[1]{\gdef\@Acknowledgments{#1}}
    
    %The abstract text
    \newcommand{\Abstract}[1]{\gdef\@Abstract{#1}}

%--------------------------------------------------------------------
% Line spacing.
%--------------------------------------------------------------------
\newlength{\defbaselineskip}
\setlength{\defbaselineskip}{\baselineskip}
\newcommand{\setlinespacing}[1]%
           {\setlength{\baselineskip}{#1 \defbaselineskip}}
\newcommand{\doublespaceing}{\setlength{\baselineskip}%
                           {2 \defbaselineskip}}
\newcommand{\singlespacing}{\setlength{\baselineskip}{\defbaselineskip}}
\setlength\arraycolsep{2pt}

%--------------------------------------------------------------------
% Set the margins.
%--------------------------------------------------------------------
% Set margins to 1 inch, including making the page number be 1 inch 
% from the bottom
\usepackage[letterpaper,margin=1in,includefoot]{geometry} 

%--------------------------------------------------------------------
% Cover Sheet.
%--------------------------------------------------------------------
\newcommand{\coversheet}{
    \newpage
    \null
    \pagestyle{empty}
}

%--------------------------------------------------------------------
% Title Page.
%--------------------------------------------------------------------
\renewcommand{\titlepage}{
    \newpage
    %set page numbering to roman, reset counting from 1, 
    %and do not display on this page
    \pagestyle{plain} 
    \thispagestyle{empty}
    \pagenumbering{roman}
    \begin{center} 
        \vspace*{0in}
        {\textbf \@titleone}\\[0.8\baselineskip]
        {\textbf \@titletwo}\\[0.8\baselineskip]
        {\textbf \@titlethree}\\
        \vskip 1in
        {\small A \@doctype}\\[0.6\baselineskip] 
        {\small Presented to the}\\[0.6\baselineskip]
        {\small Faculty of}\\[0.6\baselineskip]
        {\small California State Polytechnic University, Pomona}\\[0.6\baselineskip]
        \vskip1in
        {\small In Partial Fulfillment}\\[0.6\baselineskip]
        {\small Of the Requirements for the Degree}\\[0.6\baselineskip]
        {\small \@degree}\\[0.6\baselineskip]
        {\small In}\\[0.6\baselineskip]
        {\small \@field}\\[0.6\baselineskip]
        \vskip1in
        {\small By}\\[0.6\baselineskip]
        {\small \@Author}\\[0.6\baselineskip]
        {\small \@Year}\\[0.6\baselineskip]
    \end{center}
}

%--------------------------------------------------------------------
% Committee Membership page
%--------------------------------------------------------------------
\newcommand{\committeemembershippage}{
    \newpage
    \begin{center}   
        \vspace*{1.2cm}
        \uppercase{\textbf{Committee Membership}}\\[3\baselineskip]
    \end{center}
    \vspace{-0.2in}
    \begin{tabular}{ll}
    \textbf{\small \@doctypeUp :} & {\small \@titleone}\\
    & {\small \@titletwo}\\
    & {\small \@titlethree}\\
    &\\
    \textbf{\small AUTHOR:} & {\small \@Author}\\
    &\\
    &\\
    \textbf{\small DATE SUBMITTED:} & {\small \@Semester~\@Year}\\
    &\\
    & Department of Computer Science\\
    &\\
    &\\
    \end{tabular}
    \begin{center}
    {\small Dr.\ \@Advisor} \\
    {\small Thesis Committee Chair} \\
    {\small Associate Professor of Computer Science} \\
    \vspace*{2 \baselineskip}
    {\small Dr.\ \@MemberA} \\
    {\small Professor of Computer Science} \\
    \vspace*{2 \baselineskip}
    {\small Dr.\ \@MemberB} \\
    {\small Assistant Professor of Computer Science}
    \end{center}
}

%--------------------------------------------------------------------
% Acknowledgements page.
%--------------------------------------------------------------------
\newcommand{\acknowledgmentspage}{
    \newpage
    \begin{center}    
        \vspace*{0.375in}
        \uppercase{\textbf{Acknowledgments}}%\\[2\baselineskip]
    \end{center}
	\setlength{\baselineskip}{1.66 \defbaselineskip}
    \renewcommand{\baselinestretch}{1.66} \small\normalsize%
    \@Acknowledgments    
   	%\renewcommand{\baselinestretch}{0.6} 
}

%--------------------------------------------------------------------
% Abstract page.
%--------------------------------------------------------------------
\newcommand{\abstractpage}{
    \newpage
    \begin{center}    
        \vspace*{0.375in}
        \uppercase{\textbf{Abstract}}%\\[2\baselineskip]
    \end{center}
   \renewcommand{\baselinestretch}{1.66} \small\normalsize%
    \@Abstract
    %\renewcommand{\baselinestretch}{0.6}
}

%--------------------------------------------------------------------
% Table of Contents Adjustment.
%--------------------------------------------------------------------
% Define the command \AddChap to be issued before the first chapter 
% is defined to add the word "Chapter" to each chapter entry in the 
% ToC (as opposed to just displaying the chapter number).
% Define the command \AddAppendix to be issued before the first appendix
% is defined to add the word "Appendix" to each chapter entry in the 
% ToC (as opposed to just displaying the appendix letter).
% Define the command \SuppChap to be issued after the main chapters
% and befoe supplemental chapters such as the Bibliograph or Appendices
% so that "Chapter" is not displayed in the ToC for supplemental
% chapters.
\makeatletter
\let\orig@chapter\@chapter
\newcommand\SuppChap{%
    \let\@chapter\orig@chapter}
\newcommand\AddChap{%
    \def\@chapter[##1]##2{%
        \ifnum \c@secnumdepth > \m@ne
            \refstepcounter{chapter}%
            \typeout{\@chapapp\space\thechapter.}%
            \addcontentsline{toc}{chapter}%
                {Chapter~\protect\numberline{\thechapter}##1}%
        \else
            \addcontentsline{toc}{chapter}{##1}%
        \fi
        \chaptermark{##1}%
        \addtocontents{lof}{\protect\addvspace{10\p@}}%
        \addtocontents{lot}{\protect\addvspace{10\p@}}%
        \if@twocolumn
            \@topnewpage[\@makechapterhead{##2}]%
        \else
            \@makechapterhead{##2}%
            \@afterheading
        \fi
    }%
}
\newcommand\AddAppendix{%
    \def\@chapter[##1]##2{%
        \ifnum \c@secnumdepth > \m@ne
            \refstepcounter{chapter}%
            \typeout{\@chapapp\space\thechapter.}%
            \addcontentsline{toc}{chapter}%
                {~\protect\numberline{\thechapter}##1}%
        \else
            \addcontentsline{toc}{chapter}{##1}%
        \fi
        \chaptermark{##1}%
        \addtocontents{lof}{\protect\addvspace{10\p@}}%
        \addtocontents{lot}{\protect\addvspace{10\p@}}%
        \if@twocolumn
            \@topnewpage[\@makechapterhead{##2}]%
        \else
            \@makechapterhead{##2}%
            \@afterheading
        \fi
    }%
    \appendixtocoff%
}
\makeatother