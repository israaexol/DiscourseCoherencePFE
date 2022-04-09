import * as React from 'react'
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import "./Sidebar.css";
import { experimentalStyled as styled } from '@mui/material/styles';
import Paper from '@mui/material/Paper';
import Grid from '@mui/material/Grid';
import Link from '@mui/material/Link';

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

//const preventDefault = (event: React.SyntheticEvent) => event.preventDefault();

const Item = styled(Paper)(({ theme }) => ({
...theme.typography.body1,
padding: theme.spacing(2),
textAlign: 'center',
color: theme.palette.text.black,
width: 100,
}));

const Sidebar = () => {
  return (
      <div id="header">
        <ProSidebar>
            <SidebarContent>
                <Card sx={{ width: 350, marginTop: '5%', marginLeft: '7%' }}>
                    <CardContent>
                        <Typography sx={{ fontSize: 18, fontWeight: 'bold', fontFamily: 'Didact Gothic' }} color="#5885FB" gutterBottom>
                        Description du modèle
                        </Typography>
                        <Typography variant="body2" sx={{ fontFamily: 'Poppins', fontWeight: 300 }}>
                        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum
                        </Typography>
                    </CardContent>
                </Card>
                <Box
                    sx={{
                    display: 'flex',
                    flexDirection: 'row',
                    p: 1,
                    m: 1,
                    justifyContent: 'space-between'
                    }}
                >
                    <Item sx={{ backgroundColor: '#CADCF1', borderRadius: 2, width: '70px', height: '12px' }}>
                        <div class='parent'>
                            <div class='child'>
                                <img src={python} class='logo'></img>
                                <Typography variant="caption" sx={{ fontFamily: 'Poppins', fontWeight: 500 }}>Python</Typography>
                            </div>
                        </div>
                    </Item>
                    <Item sx={{ backgroundColor: '#FFFFFF', borderRadius: 2, width: '90px', height: '12px' }}>
                        <div class='parent'>
                            <div class='child3'>
                                <img src={scikitlearn} class='logo'></img>
                                <Typography variant="caption" sx={{ fontFamily: 'Poppins', fontWeight: 500 }}>Scikit-learn</Typography>
                            </div>
                        </div>
                    </Item>
                    <Item sx={{ backgroundColor: '#FFFFFF', borderRadius: 2, width: '70px', height: '12px' }}>
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
                    <Item sx={{ backgroundColor: '#FFFFFF', borderRadius: 2, width: '70px', height: '12px' }}>
                        <div class='parent'>
                            <div class='child'>
                                <img src={vscode} class='logo'></img>
                                <Typography variant="caption" sx={{ fontFamily: 'Poppins', fontWeight: 500 }}>VS Code</Typography>
                            </div>
                        </div>
                    </Item>
                    <Item sx={{ backgroundColor: '#FFF27B', borderRadius: 2, width: '85px', height: '12px' }}>
                        <div class='parent'>
                            <div class='child2'>
                                <img src={anaconda} class='logo2'></img>
                                <Typography variant="caption" sx={{ fontFamily: 'Poppins', fontWeight: 500 }}>Anaconda</Typography>
                            </div>
                        </div>
                    </Item>
                    <Item sx={{ backgroundColor: '#FFE1E1', borderRadius: 2, width: '70px', height: '12px' }}>
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
                        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
                        </Typography>
                    </CardContent>
                </Card>
            </SidebarContent>
            <SidebarFooter>
            <Box
            sx={{
                typography: 'body1',
                '& > :not(style) + :not(style)': {
                ml: 2,
                },
            }}
            >
                <Menu>
                    <MenuItem>
                        <img src={github} height='7%' width='7%' style={{'float': 'left', 'marginRight': '3%'}}></img>
                        <Link href="https://github.com/israaexol/DiscourseCoherencePFE"><Typography sx={{ fontSize: 18, fontWeight: 'bold', fontFamily: 'Didact Gothic' }}>Répertoire GitHub</Typography></Link>
                    </MenuItem>
                </Menu>
            </Box>
            </SidebarFooter>
        </ProSidebar>
      </div>
  )
}

export default Sidebar