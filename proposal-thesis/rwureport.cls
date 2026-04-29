\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{rwureport}[2026/02/02 Report document class for the Hochschule Ravensburg-Weingarten]
% CC-BY-SA Stephan Kleber, Stefan Diezel

% ---------------------------------------------------------------------
% Base class: article (could be changed to report if chapters are needed)
% ---------------------------------------------------------------------
\LoadClass[a4paper, 11pt, parskip=half]{scrartcl}

% ---------------------------------------------------------------------
% Page layout according to DIN 5008 (A4) + top margin = 30 mm
% ---------------------------------------------------------------------
\RequirePackage[
left=25mm,
right=20mm,
top=25mm,
bottom=25mm,
includeheadfoot
]{geometry}

\RequirePackage{rwu}

% ---------------------------------------------------------------------
% language
% ---------------------------------------------------------------------
\@ifclasswith{rwureport}{ngerman}{}{% 
    % If 'ngerman' is NOT a class option, pass USenglish to babel
    \PassOptionsToPackage{USenglish}{babel}
}
\RequirePackage{babel}

% ---------------------------------------------------------------------
% Font setup: Barlow Semi Condensed (for XeLaTeX/LuaLaTeX)
% ---------------------------------------------------------------------
\RequirePackage{iftex}
\ifXeTeX
\RequirePackage{fontspec}
%\setmainfont[
%UprightFont={BarlowSemiCondensed-Regular},
%BoldFont={BarlowSemiCondensed-Bold},
%ItalicFont={BarlowSemiCondensed-Italic},
%BoldItalicFont={BarlowSemiCondensed-Italic}
%]{Barlow Semi Condensed}
\setmainfont{Barlow Semi Condensed}
\renewcommand{\familydefault}{\sfdefault}
\setkomafont{disposition}{\normalfont\sffamily}
\else
\ifLuaTeX
\RequirePackage{fontspec}
\setmainfont[
UprightFont={BarlowSemiCondensed-Regular},
BoldFont={BarlowSemiCondensed-Bold},
ItalicFont={BarlowSemiCondensed-Italic},
BoldItalicFont={BarlowSemiCondensed-Italic}
]{Barlow Semi Condensed}
\renewcommand{\familydefault}{\sfdefault}
\setkomafont{disposition}{\normalfont\sffamily}
\else
% Fallback for pdfLaTeX
%\RequirePackage[scaled=0.95]{helvet}
\usepackage[utf8]{inputenc}
\RequirePackage{tgheros}
\renewcommand{\familydefault}{\sfdefault}
\setkomafont{disposition}{\normalfont\sffamily}
\fi
\fi

% ---------------------------------------------------------------------
% Packages for graphics and positioning
% ---------------------------------------------------------------------
\RequirePackage{graphicx}
\RequirePackage{tikz}
\RequirePackage{scrlayer-scrpage}

% ---------------------------------------------------------------------
% Logo configuration
% ---------------------------------------------------------------------
\newlength{\LogoWidth}
\newlength{\LogoHeight}
\setlength{\LogoWidth}{50mm}
\setlength{\LogoHeight}{17mm}

% Default placeholder logo (can be replaced by user)
%\newcommand{\HeaderLogo}{%
%	\begin{tikzpicture}
%		\draw[gray!60, thick] (0,0) rectangle (\LogoWidth, \LogoHeight);
%		\node[font=\sffamily\small, align=center, text=gray!70]
%		at (\LogoWidth/2, \LogoHeight/2) {RWU\\50×17 mm};
%	\end{tikzpicture}%
%}

% Replace the placeholder with your actual logo
% \renewcommand{\HeaderLogo}{\includegraphics[width=50mm,height=17mm]{rwu_logo.png}}
\newcommand{\HeaderLogo}{\resizebox{!}{17mm}{\rwulogo}}

% ---------------------------------------------------------------------
% Place logo inside top margin (not affecting text)
%   - Bottom edge: 10 mm below page top
%   - Right aligned with text block (20 mm right margin)
% ---------------------------------------------------------------------
\AddToHook{shipout/firstpage}{%  background
    \begin{tikzpicture}[remember picture, overlay]
        \coordinate (RightEdge) at ([xshift=-20mm]current page.north east);
        \node[anchor=north east, yshift=-10mm] at (RightEdge)
        {\HeaderLogo};
    \end{tikzpicture}%
}

% ---------------------------------------------------------------------
% Header & footer
% ---------------------------------------------------------------------
\pagestyle{scrheadings}
\clearpairofpagestyles
\cfoot*{\pagemark} % alternative: \cfoot[\pagemark]{\pagemark}

% ---------------------------------------------------------------------
% Section title formatting
% ---------------------------------------------------------------------
\renewcommand*{\sectionformat}{\thesection\hspace{1em}}
\renewcommand*{\subsectionformat}{\thesubsection\hspace{1em}}

% ---------------------------------------------------------------------
% Paragraph formatting
% ---------------------------------------------------------------------
\setlength{\parindent}{0pt}
\setlength{\parskip}{1ex plus 0.5ex minus 0.2ex}

% ---------------------------------------------------------------------
% End of class
% ---------------------------------------------------------------------
\endinput
