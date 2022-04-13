import * as React from 'react'
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import "./Sidebar.css";
import { experimentalStyled as styled } from '@mui/material/styles';
import Paper from '@mui/material/Paper';
import Link from '@mui/material/Link';
import DescriptionSentAvg from '../DescriptionSentAvg/DescriptionSentAvg'
import DescriptionParSeq from '../DescriptionParSeq/DescriptionParSeq'
import DescriptionSemRel from '../DescriptionSemRel/DescriptionSemRel'
import DescriptionCNNPosTag from '../DescriptionCNNPosTag/DescriptionCNNPosTag'


//import react pro sidebar components
import {
    ProSidebar,
    Menu,
    MenuItem,
    SidebarFooter,
    SidebarContent,
  } from "react-pro-sidebar";
import python from '../../assets/python.png'
import pytorch from '../../assets/pytorch.png'
import numpy from '../../assets/numpy.png'
import anaconda from '../../assets/anaconda.png'
import vscode from '../../assets/vscode.png'
import scikitlearn from '../../assets/scikitlearn.png'
import github from '../../assets/github.png'

const Item = styled(Paper)(({ theme }) => ({
...theme.typography.body1,
padding: theme.spacing(2),
textAlign: 'center',
color: theme.palette.text.black,
width: 100,
}));

function ModelDescription({niveau}) {
    switch (niveau) {
      case 0:
        return <DescriptionSentAvg/>;
      case 1:
        return <DescriptionParSeq/>;
      case 2:
        return <DescriptionSemRel/>;
      case 3:
        return <DescriptionCNNPosTag/>
      default:
        return null;
    }
  }

const Sidebar = ({selectedIndex}) => {
  return (
      <div id="header">
        <ProSidebar>
            <SidebarContent>
            { {selectedIndex} ?
                <Card sx={{ width: 350, marginTop: '5%', marginLeft: '7%' }}>
                    <CardContent>
                        <Typography sx={{ fontSize: 18, fontWeight: 'bold', fontFamily: 'Didact Gothic' }} color="#5885FB" gutterBottom>
                        Description du modèle
                        </Typography>
                        <ModelDescription niveau={selectedIndex}/>
                    </CardContent>
                </Card> : <></>
            }
                <Typography sx={{ fontSize: 18, fontWeight: 'bold', fontFamily: 'Didact Gothic', marginLeft: '5%', marginTop: '4%' }} color="#5885FB" gutterBottom>
                        Technologies, librairies et outils utilisés
                </Typography>
                <Box
                    sx={{
                    display: 'flex',
                    flexDirection: 'row',
                    p: 1,
                    m: 1,
                    justifyContent: 'space-between'
                    }}
                >
                    <Item sx={{ backgroundColor: '#CADCF1', borderRadius: 2, width: '95px', height: '12px' }}>
                        <div class='parent'>
                            <div class='child'>
                                <img src={python} class='logo'></img>
                                <Typography variant="caption" sx={{ fontFamily: 'Poppins', fontWeight: 500 }}>Python</Typography>
                            </div>
                        </div>
                    </Item>
                    <Item sx={{ backgroundColor: '#FFFFFF', borderRadius: 2, width: '125px', height: '12px' }}>
                        <div class='parent'>
                            <div class='child3'>
                                <img src={scikitlearn} class='logo'></img>
                                <Typography variant="caption" sx={{ fontFamily: 'Poppins', fontWeight: 500 }}>Scikit-learn</Typography>
                            </div>
                        </div>
                    </Item>
                    <Item sx={{ backgroundColor: '#FFFFFF', borderRadius: 2, width: '95px', height: '12px' }}>
                        <div class='parent'>
                            <div class='child'>
                                <img src={numpy} class='logo'></img>
                                <Typography variant="caption" sx={{ fontFamily: 'Poppins', fontWeight: 500 }}>Numpy</Typography>
                            </div>
                        </div>
                    </Item>
                </Box>
                <Box
                    sx={{
                    display: 'flex',
                    flexDirection: 'row',
                    p: 1,
                    m: 1,
                    justifyContent: 'space-between',
                    alignItems: 'flex-start'
                    }}
                >
                    <Item sx={{ backgroundColor: '#FFFFFF', borderRadius: 2, width: '95px', height: '12px' }}>
                        <div class='parent'>
                            <div class='child'>
                                <img src={vscode} class='logo'></img>
                                <Typography variant="caption" sx={{ fontFamily: 'Poppins', fontWeight: 500 }}>VS Code</Typography>
                            </div>
                        </div>
                    </Item>
                    <Item sx={{ backgroundColor: '#FFF27B', borderRadius: 2, width: '125px', height: '12px' }}>
                        <div class='parent'>
                            <div class='child2'>
                                <img src={anaconda} class='logo2'></img>
                                <Typography variant="caption" sx={{ fontFamily: 'Poppins', fontWeight: 500 }}>Anaconda</Typography>
                            </div>
                        </div>
                    </Item>
                    <Item sx={{ backgroundColor: '#FFE1E1', borderRadius: 2, width: '95px', height: '12px' }}>
                        <div class='parent'>
                            <div class='child'>                            
                                <img class='logo' src={pytorch}></img>
                                <Typography variant="caption" sx={{ fontFamily: 'Poppins', fontWeight: 500 }}>Pytorch</Typography>
                            </div>
                        </div>
                    </Item>
                    
                </Box>
                <Card sx={{ width: 350, marginTop: '5%', marginLeft: '7%' }}>
                    <CardContent>
                        <Typography sx={{ fontSize: 18, fontWeight: 'bold', fontFamily: 'Didact Gothic' }} color="#5885FB" gutterBottom>
                        Dataset utilisé
                        </Typography>
                        <Typography variant="body2" sx={{ fontFamily: 'Poppins', fontWeight: 300 }}>
                            <b>GCDC</b> <Link href="https://github.com/aylai/GCDC-corpus" sx={{fontWeight : 500}}>(Grammarly Corpus of Discourse Coherence)</Link>
                            est un corpus introduit par <i>(Lai et Tetreault, 2018)</i> composé d'un total de 4800 documents provenant
                            de 4 différents corpus, chacun relatif à un domaine (Emails d'<b>Enron</b>, Emails de <b>Clinton</b>, Réponses de forum <b>Yahoo</b> et Revues de business <b>Yelp</b>)

                        </Typography>
                    </CardContent>
                </Card>
            </SidebarContent>
            <SidebarFooter>
            <Box
            sx={{
                display: 'flex',
                flexDirection: 'row',
                p: 1,
                m: 1,
                justifyContent: 'flex-start',
                alignItems: 'flex-start'
            }}
            >
                <div class='parent'>
                    <div class='child3'></div>
                        <img src={github} height='7%' width='7%' class="logo3"></img>
                        <Link href="https://github.com/israaexol/DiscourseCoherencePFE"><Typography sx={{ fontSize: 18, fontWeight: 'bold', fontFamily: 'Didact Gothic', marginLeft: '12px' }}>Répertoire GitHub</Typography></Link>
                    </div>
                <div/>
            </Box>
            </SidebarFooter>
        </ProSidebar>
      </div>
  )
}

export default Sidebar