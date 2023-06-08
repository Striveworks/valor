import { useAuth0 } from '@auth0/auth0-react';
import Typography from '@mui/material/Typography';
import { usingAuth } from './auth';
import { Wrapper } from './components/wrapper';
import { LoginButton } from './login-button';

export const HomePage = () => {
	const { isAuthenticated } = useAuth0();

	const content = (
		<div className='App'>
			<header className='App-header'>
				<Typography variant='h1'>velour</Typography>
				<br />
				{usingAuth() && !isAuthenticated ? <LoginButton /> : <></>}
			</header>
		</div>
	);

	if (!usingAuth() || isAuthenticated) {
		return <Wrapper>{content}</Wrapper>;
	}
	return content;
};
